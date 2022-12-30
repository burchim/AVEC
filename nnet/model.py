# Copyright 2021, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Other
from tqdm import tqdm
import os
import time
import copy
import matplotlib.pyplot as plt
import glob

# Dictionaries
from nnet.optimizers import optim_dict
from nnet.losses import loss_dict
from nnet.decoders import decoder_dict
from nnet.metrics import metric_dict

# Neural Nets
from nnet.module import Module
from nnet import normalizations as norms

from nnet.schedulers import (
    Scheduler,
    ConstantScheduler
)

class Model(Module):

    def __init__(self, name="model"):
        super(Model, self).__init__()

        # Model Attributes
        self.is_distributed = False
        self.rank = 0
        self.is_parallel = False
        self.compiled = False
        self.built = False
        self.name = name
        self.ema_model = None
        self.ema_tau = 0.0
        self.grad_max_norm = None

    def distribute_strategy(self, rank, sync_batch_norm=True):
        if sync_batch_norm:
            object.__setattr__(self, "ddp", torch.nn.parallel.DistributedDataParallel(norms.SyncBatchNorm.convert_sync_batchnorm(self), device_ids=[rank]))
        else:
            object.__setattr__(self, "ddp", torch.nn.parallel.DistributedDataParallel(self, device_ids=[rank]))
        self.rank = rank
        self.is_distributed = True

    def parallel_strategy(self):
        object.__setattr__(self, "dp", torch.nn.DataParallel(self))
        self.is_parallel = True

    def set_ema(self, ema_tau):

        """ Exponential Moving Average Model """

        object.__setattr__(self, "ema_model", copy.deepcopy(self))
        self.set_require_grad(self.ema_model, False)
        self.ema_model.eval()
        self.ema_tau = ema_tau

    def compile(self, losses, loss_weights=None, optimizer="Adam", metrics=None, decoders=None):

        # Optimizer
        if isinstance(optimizer, str):
            self.optimizer = optim_dict[optimizer](params=self.parameters())
        else:
            self.optimizer = optimizer

        # Model Step
        self.model_step = self.optimizer.scheduler.model_step

        # Losses
        if isinstance(losses, str):
            self.compiled_losses = loss_dict[losses]()
        elif losses == None:
            self.compiled_losses = []
        else:
            self.compiled_losses = losses

        # Loss Weights
        if loss_weights == None:

            self.compiled_loss_weights = ConstantScheduler(1.0)

        elif isinstance(loss_weights, float):

            self.compiled_loss_weights = ConstantScheduler(loss_weights)

        else:

            # Assert List or Dict
            assert isinstance(loss_weights, dict) or isinstance(loss_weights, list)

            # Convert to Scheduler
            if isinstance(loss_weights, dict):
                for key, value in loss_weights.items():
                    if not isinstance(value, Scheduler):
                        loss_weights[key] = ConstantScheduler(value)
            else:
                for i, value in enumerate(loss_weights):
                    if not isinstance(value, Scheduler):
                        loss_weights[i] = ConstantScheduler(value)

            # Assign
            self.compiled_loss_weights = loss_weights

        # Metrics
        if isinstance(metrics, str):
            self.compiled_metrics = metric_dict[metrics]()
        elif metrics == None:
            self.compiled_metrics = []
        else:
            self.compiled_metrics = metrics
            
        # Decoders
        if isinstance(decoders, str):
            self.compiled_decoders = decoder_dict[decoders]()
        elif decoders == None:
            self.compiled_decoders = []
        else:
            self.compiled_decoders = decoders

        # Set Compiled to True
        self.compiled = True

    def build(self, outputs):

        # Map to Outputs
        self.losses = self.map_to_outputs(outputs, self.compiled_losses)
        self.loss_weights = self.map_to_outputs(outputs, self.compiled_loss_weights)
        self.decoders = self.map_to_outputs(outputs, self.compiled_decoders)
        self.metrics = self.map_to_outputs(outputs, self.compiled_metrics)

        # Transfer to Device
        self.losses = self.transfer_to_device(self.losses)
        self.decoders = self.transfer_to_device(self.decoders)
        self.metrics = self.transfer_to_device(self.metrics)

        # Build Ema
        if self.ema_model != None:
            self.ema_model.losses = self.losses
            self.ema_model.loss_weights = self.loss_weights
            self.ema_model.decoders = self.decoders
            self.ema_model.metrics = self.metrics
            self.ema_model.built = True

        # Set Built to true
        self.built = True

        # Print
        if self.rank == 0:
            print("Built", self.name)
            print("losses:", {key:type(value).__name__ for key, value in self.losses.items()})
            print("loss weights:", {key:type(value).__name__ for key, value in self.loss_weights.items()})
            print("metrics:", {key:type(value).__name__ for key, value in self.metrics.items()})
            print("decoders:", {key:type(value).__name__ for key, value in self.decoders.items()})

    def map_to_outputs(self, outputs, struct):

        """Convenience method to conform `struct` to `outputs` structure.

        Mappings performed:
            (1) Map a struct to a dict of outputs, using the output names.
            (2) Fill missing struct elements with None.
            (3) Map a single item to all outputs.

        Args:
            outputs: Model outputs predictions dict.
            struct: Arbitrary nested structure (dict, list, item).

        Returns:
            Dict mapping `struct` to `outputs` structure.

        """

        # None
        if struct == None:

            return struct

        # Dictionary
        elif isinstance(struct, dict):

            # Assert struct key in outputs
            for key in struct:
                if not key in outputs:
                    raise Exception("Found unexpected dict key: {}. Valid output names are: {}".format(key, outputs.keys()))

            # Fill missing key with None
            for key in outputs:
                if not key in struct:
                    struct[key] = None

        # List
        elif isinstance(struct, list):

            # Map list items to outputs, Fill missing items with None, Ignore extra items
            struct = {key: struct[i] if i < len(struct) else None for i, key in enumerate(outputs)}

        # Module / Tensor / tuple
        else:

            # Map item to all outputs
            struct = {key: struct for key in outputs}

        return struct

    def forward_model(self, inputs, targets, compute_metrics=True, verbose=0):

        """ forward_model method

        - forward
        - compute losses
        - compute metrics
        
        """

        # Init Batch Dict
        batch_losses = {}
        batch_metrics = {}
        batch_truths = {}
        batch_preds = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # Additional Targets
        self.additional_targets = {}

        # Forward
        if self.is_distributed:
            outputs = self.ddp(inputs)
        elif self.is_parallel:
            outputs = self.dp(inputs)
        else:
            outputs = self.forward(inputs)

        # Format Outputs to dict
        if isinstance(outputs, dict):
            pass
        elif isinstance(outputs, list):
            outputs = {"output_" + str(key): value for key, value in enumerate(outputs)}
        else:
            outputs = {"output": outputs}

        # Map Targets to Outputs
        targets = self.map_to_outputs(outputs, targets)

        # Append Additional Targets
        for key in self.additional_targets:
            targets[key] = self.additional_targets[key]

        # Build Model
        if not self.built:
            self.build(outputs)

        # Outputs loop
        for key in outputs:

            # Loss Function
            if self.losses[key] != None:

                # Loss key
                key_loss = "loss_" + key

                # Loss
                batch_losses[key_loss] = self.losses[key](targets[key], outputs[key])

                # Weight Loss
                total_loss += batch_losses[key_loss] * self.loss_weights[key].get_val_step(self.model_step + 1)

            # Metric Functions
            if self.metrics[key] != None and compute_metrics:

                # To list
                if not isinstance(self.metrics[key], list):
                    metrics = [self.metrics[key]]
                else:
                    metrics = self.metrics[key]
                if not isinstance(self.decoders[key], list):
                    decoders = [self.decoders[key] for _ in metrics]
                else:
                    decoders = self.decoders[key]


                for metric, decoder in zip(metrics, decoders):

                    # Metric Key
                    key_metric = metric.name
                    if key_metric in batch_metrics:
                        key_metric += "_" + key

                    # Decoding
                    if decoder != None:
                        batch_truths[key_metric] = decoder(targets[key], from_logits=False) if targets[key] != None else None
                        batch_preds[key_metric] = decoder(outputs[key])
                    else:
                        batch_truths[key_metric] = targets[key]
                        batch_preds[key_metric] = outputs[key]

                    # Prediction Verbose
                    if verbose:
                        print("Groundtruths:\n", batch_truths[key_metric])
                        print("Predictions:\n", batch_preds[key_metric])

                    # Metric
                    batch_metrics[key_metric] = metric(batch_truths[key_metric], batch_preds[key_metric])

        # Module Infos / Losses
        for module in self.modules():
            if hasattr(module, "added_losses"):# and module is not self:
                for key, value in module.added_losses.items():
                    key_loss = "loss_" + key
                    batch_losses[key_loss] = value["loss"]
                    total_loss += batch_losses[key_loss] * value["weight"]
                module.reset_losses()
            if hasattr(module, "infos") and module is not self:
                self.infos.update(module.infos)
                module.reset_infos()

        # Append Total loss
        if len(batch_losses) > 1:
            batch_losses = dict({"loss": total_loss}, **batch_losses)
        else:
            batch_losses = {"loss": total_loss}

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def train_step(self, inputs, targets, precision, grad_scaler, accumulated_steps, acc_step, eval_training):

        """ train_step method

        - forward_model (forward + compute losses/metrics)
        - backward
        
        """

        # Automatic Mixed Precision Casting (model forward + loss computing)
        if "cuda" in str(self.device):
            with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)
        else:
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, compute_metrics=eval_training)

        # Accumulated Steps
        loss = batch_losses["loss"] / accumulated_steps
        acc_step += 1

        # Backward: Accumulate gradients
        grad_scaler.scale(loss).backward()

        # Continue Accumulating
        if acc_step < accumulated_steps:
            return batch_losses, batch_metrics, acc_step

        # Grad Scaler Info
        if grad_scaler.is_enabled():
            self.add_info("grad_scale", grad_scaler.get_scale())

        # Unscale Gradients
        grad_scaler.unscale_(self.optimizer)

        # Clip Gradients Global Norm
        if self.grad_max_norm != None:
            if not sum(v.item() for v in grad_scaler._found_inf_per_device(self.optimizer).values()): # nan grad
                self.add_info("grad_norm", round(torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_max_norm).item(), 4))

        # Optimizer Step and Update Scale
        grad_scaler.step(self.optimizer)
        grad_scaler.update()

        # Zero Gradients
        self.optimizer.zero_grad()
        acc_step = 0

        # Update Model Infos
        self.add_info("lr", float(self.optimizer.param_groups[0]['lr']))
        self.add_info("step", self.model_step.item())

        # Cuda Infos
        if "cuda" in str(self.device) and self.model_step % 100 == 0:
            self.add_info("{}_memory".format(str(self.device)), round(100 * torch.cuda.memory_reserved(self.device) / torch.cuda.get_device_properties(self.device).total_memory, 2))

        # Update Exp Moving Avg Model
        if self.ema_model != None:
            for param_target, param_net in zip(self.ema_model.parameters(), self.parameters()):
                param_target.mul_(self.ema_tau)
                param_target.add_((1 - self.ema_tau) * param_net.detach())
            for buffer_target, buffer_net in zip(self.ema_model.buffers(), self.buffers()):
                buffer_target.copy_(buffer_net)

        return batch_losses, batch_metrics, acc_step  

    def eval_step(self, inputs, targets, verbose=0):

        with torch.no_grad():
            batch_losses, batch_metrics, batch_truths, batch_preds = self.forward_model(inputs, targets, verbose=verbose)

        return batch_losses, batch_metrics, batch_truths, batch_preds

    def num_params(self, module=None):

        if module != None:
            if isinstance(module, list):
                return sum([self.num_params(m) for m in module])
            else:
                return sum([p.numel() for p in module.parameters()])
        else:
            return sum([p.numel() for p in self.parameters()])

    def summary(self, show_dict=False, show_modules=False, plot_params=False):

        # Model Name
        print("Model name: {}".format(self.name))

        # Number Params
        print("Number Parameters: {:,}".format(self.num_params()))

        # Options
        if show_dict:
            self.show_dict()
        if show_modules:
            self.show_modules()
        if plot_params:
            self.plot_params()

        # Modules Buffer
        for key, value in self.modules_buffer.items():
            print("{} Parameters: {:,}".format(key, value.num_params()))
            if show_dict:
                value.show_dict()
            if show_modules:
                value.show_modules()

    def show_dict(self, module=None):

        # Print
        print("State Dict:")

        # Default Dict
        if module != None:
            state_dict = module.state_dict(keep_vars=True)
        else:
            state_dict = self.state_dict(keep_vars=True)

        # Empty Dict
        if state_dict == {}:
            return

        # Show Dict
        max_len_id = len(str(len(state_dict)))
        max_len_key = max([len(key) for key in state_dict.keys()]) + 5
        for id, (key, value) in enumerate(state_dict.items()):
            print("{} {} type: {:<12} numel: {:<12} shape: {:<20} mean: {:<12.4f} std: {:<12.4f} min: {:<12.4f} max: {:<12.4f} dtype: {:<12} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), "param" if isinstance(value, nn.Parameter) else "buffer", value.numel(), str(tuple(value.size())), value.float().mean(), value.float().std(), value.float().min(), value.float().max(), str(value.dtype).replace("torch.", ""), str(value.device)))

    def show_modules(self, module=None):

        # Print
        print("Named Modules:")

        # Named Modules
        if module != None:
            named_modules = dict(module.named_modules())
        else:
            named_modules = dict(self.named_modules())

        # Show Modules
        max_len_id = len(str(len(named_modules)))
        max_len_key = max([len(key) for key in named_modules.keys()]) + 5
        max_len_class = max([len(type(value).__name__) for value in named_modules.values()]) + 5
        for id, (key, value) in enumerate(named_modules.items()):
            print("{} {} class: {} device: {}".format(str(id) + " " * (max_len_id - len(str(id))), key + " " * (max_len_key - len(key)), type(value).__name__ + " " * (max_len_class - len(type(value).__name__)), value.device if hasattr(value, "device") else ""))

    def plot_params(self):

        for param_name, param in self.named_parameters():

            plt.title("{} \n shape: {} / mean: {:.4f} / std: {:.4f}".format(param_name, str(tuple(param.size())), param.mean(), param.std()))
            plt.hist(param.detach().numpy().flatten(), bins=100, density=True)
            plt.show()

    def save(self, path, save_optimizer=True):
        
        # Save Model Checkpoint
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": None if not save_optimizer else {key: value.state_dict() for key, value in self.optimizer.items()} if isinstance(self.optimizer, dict) else self.optimizer.state_dict(),
            "model_step": self.model_step,
            "is_distributed": self.is_distributed or self.is_parallel,
            "ema_model_state_dict": None if self.ema_model == None else self.ema_model.state_dict(),
            "grad_scaler_state_dict": self.grad_scaler.state_dict() if hasattr(self, "grad_scaler") else None
            }, path)

        # Print Model state
        if self.rank == 0:
            print("Model saved at step {}".format(self.model_step))

    def load(self, path, load_optimizer=True, verbose=True, strict=True):

        # Load Model Checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Load Model State Dict
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.load_state_dict({key.replace(".module.", "."):value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)
        else:
            self.load_state_dict({key:value for key, value in checkpoint["model_state_dict"].items()}, strict=strict)

        # Load Optimizer State Dict
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:

            if isinstance(self.optimizer, dict):
                for key, value in self.optimizer.items():
                    value.load_state_dict(checkpoint["optimizer_state_dict"][key])
            else:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Model Step
            self.model_step.fill_(checkpoint["model_step"])

        # Load EMA Model State Dict
        if checkpoint["ema_model_state_dict"] is not None:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

        # Print Model state
        if self.rank == 0 and verbose:
            print("Rank {}: Model loaded at step {}".format(self.rank, self.model_step))

    def on_epoch_end(self, evaluate, save, log_figure, callback_path, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics):
        self.on_step_end(evaluate, save, log_figure, callback_path, epoch, epoch, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, tag="epoch")

        # Print
        if self.rank == 0:
            print()

    def on_step_end(self, evaluate, save, log_figure, callback_path, epoch, step, inputs, targets, dataset_eval, eval_steps, verbose_eval, writer, recompute_metrics, tag="step"):

        # Evaluate Model
        if evaluate:
            self._evaluate(dataset_eval, writer, step, eval_steps, verbose_eval, recompute_metrics, tag="Evaluation-" + tag)
            self.train()

        # Save Checkpoint
        if save and callback_path and self.rank == 0:
            self.save(os.path.join(callback_path, "checkpoints_epoch_{}_step_{}.ckpt".format(epoch, self.model_step)))

        # Log Figure
        if log_figure and callback_path:
            self.eval()
            self.log_figure(step, inputs, targets, writer, tag)
            self.train()

    def log_figure(self, step, inputs, targets, writer, tag): 
        pass

    def display_step(self, losses, metrics, infos, epoch_iterator, step):

        # Description
        description = ""

        # Losses
        for key, value in losses.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Metrics
        for key, value in metrics.items():
            description += "{}: {:.4f} - ".format(key, value / step)

        # Infos
        for key, value in infos.items():
            if key.endswith("lr"):
                description += "{}: {:.2e} - ".format(key, value)
            else:
                description += "{}: {} - ".format(key, value)

        # Set description
        epoch_iterator.set_description(description)

    def log_step(self, losses, metrics, infos, writer, step, tag, dist_log=False):

        # Update tag if dist log
        if dist_log:
            tag = tag + "-" + str(self.rank)

        # Log Step
        if dist_log or self.rank == 0: 

            # Losses
            for key, value in losses.items():
                writer.add_scalar(os.path.join(tag, key), value, step)

            # Metrics
            for key, value in metrics.items():
                writer.add_scalar(os.path.join(tag, key), value, step)

            # Infos
            for key, value in infos.items():
                if isinstance(value, float) or isinstance(value, int):
                    writer.add_scalar(os.path.join(tag, key), float(value), step)

    def print_step(self, losses, metrics, tag):

        # Losses
        for key, value in losses.items():
            print("{} {}: {:.4f}".format(tag, key, value))

        # val metrics
        for key, value in metrics.items():
            print("{} {}: {:.4f}".format(tag, key, value))

    def reduce_losses_metrics(self, losses, metrics):

        # Process Barrier
        torch.distributed.barrier()

        # Losses
        for key, value in losses.items():
            torch.distributed.all_reduce(value)
            losses[key] = value / torch.distributed.get_world_size()

        # Epoch Metrics
        for key, value in metrics.items():
            torch.distributed.all_reduce(value)
            metrics[key] = value / torch.distributed.get_world_size()

        return losses, metrics

    def gather_truths_preds(self, truths, preds):

        # Process Barrier
        torch.distributed.barrier()

        # All Gather Truths
        for key, value in truths.items():
            value_gather = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(value_gather, value)
            truths[key] = []
            for val in value_gather:
                truths[key] += val

        # All Gather Preds
        for key, value in preds.items():
            value_gather = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(value_gather, value)
            preds[key] = []
            for val in value_gather:
                preds[key] += val

        return truths, preds

    def fit(self, dataset_train, epochs, dataset_eval=None, eval_steps=None, verbose_eval=0, initial_epoch=0, callback_path=None, steps_per_epoch=None, precision=torch.float32, accumulated_steps=1, eval_period_step=None, eval_period_epoch=1, saving_period_step=None, saving_period_epoch=1, log_figure_period_step=None, log_figure_period_epoch=1, step_log_period=10, eval_training=True, dist_log=False, grad_init_scale=65536.0, detect_anomaly=False, recompute_metrics=False):

        # Is Compiled
        if not self.compiled:
            if self.is_distributed:
                torch.distributed.destroy_process_group()
            raise Exception("You must compile your model before training/testing.")

        # Mixed Precision Gradient Scaler
        self.grad_scaler = torch.cuda.amp.GradScaler(init_scale=grad_init_scale, enabled=(grad_init_scale != None) and (precision==torch.float16) and ("cuda" in str(self.device)))

        # Anomaly Enabled
        torch.set_anomaly_enabled(detect_anomaly)

        # Init Training
        acc_step = 0

        # Zero Gradients
        self.zero_grad()

        # Callbacks
        if (dist_log or self.rank == 0) and callback_path is not None:

            # Create Callback Dir
            if not os.path.isdir(callback_path) and self.rank == 0:
                os.makedirs(callback_path, exist_ok=True)

            # Create Writer
            writer = SummaryWriter(os.path.join(callback_path, "logs"))

        else:

            writer = None

        # Try Catch
        try:

            # Training Loop
            for epoch in range(initial_epoch, epochs):

                # Sync sampler if distributed
                if self.is_distributed:
                    dataset_train.sampler.set_epoch(epoch)

                # Init Iterator
                if self.rank == 0:
                    print("Epoch {}/{}:".format(epoch + 1, epochs))
                    epoch_iterator = tqdm(dataset_train, total=steps_per_epoch * accumulated_steps if steps_per_epoch else None, dynamic_ncols=True)
                else:
                    epoch_iterator = dataset_train

                # Init Epoch Dict
                epoch_losses = {}
                epoch_metrics = {}

                # Clear Infos
                self.reset_infos()

                # Training Mode
                self.train()

                # Epoch training loop
                for step, batch in enumerate(epoch_iterator):

                    # Clear display bar before build
                    if not self.built and self.rank == 0:
                        epoch_iterator.clear()

                    # Unpack Batch
                    inputs, targets = batch["inputs"], batch["targets"]

                    # Transfer Batch elt to model device
                    inputs = self.transfer_to_device(inputs)
                    targets = self.transfer_to_device(targets)

                    # Train Step
                    #with torch.autograd.set_detect_anomaly(True):
                    batch_losses, batch_metrics, acc_step = self.train_step(inputs=inputs, targets=targets, precision=precision, grad_scaler=self.grad_scaler, accumulated_steps=accumulated_steps, acc_step=acc_step, eval_training=eval_training)

                    # Update Epoch Loss and Metric
                    for key, value in batch_losses.items():
                        epoch_losses[key] = epoch_losses[key] + value.detach() if key in epoch_losses else value.detach().type(torch.float64)
                    for key, value in batch_metrics.items():
                        epoch_metrics[key] = epoch_metrics[key] + value.detach() if key in epoch_metrics else value.detach().type(torch.float64)

                    # Continue Accumulating
                    if acc_step > 0:
                        continue

                    # Step Print
                    if self.rank == 0:
                        self.display_step(epoch_losses, epoch_metrics, self.infos, epoch_iterator, step + 1)

                    # Logs Step
                    if writer is not None and self.model_step % step_log_period == 0:
                        self.log_step(batch_losses, batch_metrics, self.infos, writer, self.model_step, "Training-step", dist_log)

                    # On Batch End
                    self.on_step_end(
                        evaluate=self.model_step % eval_period_step == 0 if eval_period_step != None else False,
                        save=self.model_step % saving_period_step == 0 if saving_period_step != None else False, 
                        log_figure=self.model_step % log_figure_period_step == 0 if log_figure_period_step != None else False, 
                        callback_path=callback_path, 
                        epoch=epoch + 1,
                        step=self.model_step, 
                        inputs=inputs, 
                        targets=targets, 
                        dataset_eval=dataset_eval, 
                        eval_steps=eval_steps, 
                        verbose_eval=verbose_eval,
                        writer=writer,
                        recompute_metrics=recompute_metrics
                    )

                    # Step per Epoch
                    if steps_per_epoch is not None:
                        if step + 1 >= steps_per_epoch * accumulated_steps:
                            break

                # Reduce among devices
                if self.is_distributed:
                    epoch_losses, epoch_metrics = self.reduce_losses_metrics(epoch_losses, epoch_metrics)

                # Mean loss
                for key, value in epoch_losses.items():
                    epoch_losses[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Mean Metrics
                for key, value in epoch_metrics.items():
                    epoch_metrics[key] = value / (steps_per_epoch * accumulated_steps if steps_per_epoch is not None else len(dataset_train))

                # Logs Epoch
                if self.rank == 0 and writer is not None:
                    self.log_step(epoch_losses, epoch_metrics, {}, writer, epoch + 1, "Training-epoch")

                # On Epoch End
                self.on_epoch_end(
                    evaluate=(epoch + 1) % eval_period_epoch == 0 if eval_period_epoch != None else False,
                    save=(epoch + 1) % saving_period_epoch == 0 if saving_period_epoch != None else False, 
                    log_figure=(epoch + 1) % log_figure_period_epoch == 0 if log_figure_period_epoch != None else False, 
                    callback_path=callback_path, 
                    epoch=epoch + 1, 
                    inputs=inputs, 
                    targets=targets, 
                    dataset_eval=dataset_eval, 
                    eval_steps=eval_steps, 
                    verbose_eval=verbose_eval,
                    writer=writer,
                    recompute_metrics=recompute_metrics
                )

        # Exception Handler
        except Exception as e:

            if self.is_distributed:
                torch.distributed.destroy_process_group()

            if self.rank == 0 and writer is not None:
                writer.add_text("Exceptions", "Rank: {} \nDate: {} \n{}".format(str(self.rank), time.ctime(), str(e)), self.model_step)

            raise e

    def _evaluate(self, dataset, writer, step, eval_steps=None, verbose=0, recompute_metrics=False, tag="Evaluation"):
        
        # Evaluation Dataset
        if dataset:

            # Dataset to list
            if not isinstance(dataset, list):
                dataset = [dataset]

            # Eval Datasets loop
            for dataset_i, dataset in enumerate(dataset):

                # Evaluate
                val_losses, val_metrics = self.evaluate(dataset, eval_steps, verbose, recompute_metrics)

                # Print
                if self.rank == 0:
                    self.print_step(val_losses, val_metrics, "eval")

                # Log
                if self.rank == 0 and writer is not None:
                    self.log_step(val_losses, val_metrics, {}, writer, step, os.path.join(tag, str(dataset_i)))

                # Evaluate EMA model
                if self.ema_model != None:

                    # Evaluate
                    val_losses, val_metrics = self.ema_model.evaluate(dataset, eval_steps, verbose)

                    # Print
                    if self.rank == 0:
                        self.print_step(val_losses, val_metrics, "ema eval")

                    # Log
                    if self.rank == 0 and writer is not None:
                        self.log_step(val_losses, val_metrics, {}, writer, step, os.path.join(tag + "-ema", str(dataset_i)))

    def evaluate(self, dataset_eval, eval_steps=None, verbose=0, recompute_metrics=False):

        # Evaluzation Mode
        self.eval()

        # Clear Infos
        self.reset_infos()

        # Init Epoch Dict
        epoch_losses = {}
        epoch_metrics = {}
        if recompute_metrics:
            epoch_truths = {}
            epoch_preds = {}

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval, total=eval_steps, dynamic_ncols=True)
        else: 
            eval_iterator = dataset_eval

        # Evaluation Loop
        for step, batch in enumerate(eval_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)
            targets = self.transfer_to_device(targets)

            # Eval Step
            batch_losses, batch_metrics, batch_truths, batch_preds = self.eval_step(inputs, targets, verbose)

            # Update Epoch Dict
            for key, value in batch_losses.items():
                epoch_losses[key] = epoch_losses[key] + value if key in epoch_losses else value.type(torch.float64)
            for key, value in batch_metrics.items():
                epoch_metrics[key] = epoch_metrics[key] + value if key in epoch_metrics else value.type(torch.float64)
            if recompute_metrics:
                for key, value in batch_truths.items():
                    epoch_truths[key] = epoch_truths[key] + value if key in epoch_truths else value
                for key, value in batch_preds.items():
                    epoch_preds[key] = epoch_preds[key] + value if key in epoch_preds else value

            # Step print (Rank 0)
            if self.rank == 0:
                self.display_step(epoch_losses, epoch_metrics, self.infos, eval_iterator, step + 1)

            # Evaluation Steps
            if eval_steps:
                if step + 1 >= eval_steps:
                    break

        # Reduce among devices
        if self.is_distributed:
            if recompute_metrics:
                epoch_losses, _ = self.reduce_losses_metrics(epoch_losses, {})
                epoch_truths, epoch_preds = self.gather_truths_preds(epoch_truths, epoch_preds)
            else:
                epoch_losses, epoch_metrics = self.reduce_losses_metrics(epoch_losses, epoch_metrics)

        # Mean loss
        for key, value in epoch_losses.items():
            epoch_losses[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        # Recompute Metrics
        if recompute_metrics:
            for key in epoch_metrics.keys():
                epoch_metrics[key] = self.metrics["outputs"](epoch_truths[key], epoch_preds[key]) # fix metrics key
        # Mean Metrics
        else:
            for key, value in epoch_metrics.items():
                epoch_metrics[key] = value / (eval_steps if eval_steps is not None else len(dataset_eval))

        return epoch_losses, epoch_metrics

    def swa(self, dataset, callback_path, start_epoch, end_epoch, epochs_list=None, update_steps=None, swa_type="equal", swa_decay=0.9, precision=torch.float32):

        # Create SWA Model
        if swa_type == "equal":
            swa_model = torch.optim.swa_utils.AveragedModel(self)
        elif swa_type == "exp":
            swa_model = torch.optim.swa_utils.AveragedModel(self, avg_fn=lambda averaged_model_parameter, model_parameter, num_averaged: (1 - swa_decay) * averaged_model_parameter + swa_decay * model_parameter)

        if self.rank == 0:
            if epochs_list:
                print("Stochastic Weight Averaging on checkpoints : {}".format(epochs_list))
            else:
                print("Stochastic Weight Averaging on checkpoints : {}-{}".format(start_epoch, end_epoch))

        # Update SWA Model Params
        if epochs_list == None:
            epochs_list = list(range(int(start_epoch), int(end_epoch) + 1))
        for epoch in epochs_list:

            # Load Model Checkpoint
            self.load(glob.glob(os.path.join(callback_path, "checkpoints_epoch_{}_step_*.ckpt".format(epoch)))[0])

            # Update SWA Model
            swa_model.update_parameters(self)

        # Load SWA Model Params
        self.load_state_dict({key[7:]:value for key, value in swa_model.state_dict().items() if key != "n_averaged"})

        if self.rank == 0:
            print("Updating Batch Normalization Statistics")

        # Number Update Steps
        steps = 0
        if update_steps == None:
            update_steps = len(dataset)

        # Init
        self.train()
        if self.rank == 0:
            p_bar = tqdm(range(update_steps))
        else:
            p_bar = None

        # Update Batch Normalization Statistics
        while steps < update_steps:
            for step, batch in enumerate(dataset):

                # Unpack Batch
                inputs = batch["inputs"]

                # Transfer Batch elt to model device
                inputs = self.transfer_to_device(inputs)

                # Forward
                with torch.cuda.amp.autocast(enabled=precision!=torch.float32, dtype=precision):
                    with torch.no_grad():
                        self.forward(inputs)

                # update_steps
                steps += 1
                if p_bar is not None: 
                    p_bar.update(1)
                if steps == update_steps:
                    break

        # Save Model
        if self.rank == 0:
            self.save(os.path.join(callback_path, "checkpoints_swa-{}-{}-{}.ckpt".format(swa_type, epochs_list[0], epochs_list[-1])), save_optimizer=False)

        # Barrier
        if self.is_distributed:
            torch.distributed.barrier()

    def generate(self, dataset, saving_path=None):

        # Eval mode
        self.eval()

        # Create Saving Path
        if saving_path != None and self.rank == 0:
            if not os.path.isdir(saving_path):
                os.makedirs(saving_path)

        # Init
        if self.rank == 0:
            epoch_iterator = tqdm(dataset, dynamic_ncols=True)
        else:
            epoch_iterator = dataset

        # Epoch training loop
        ctr = 0
        for step, batch in enumerate(epoch_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)

            # Generate Samples
            self.forward_generate(inputs, saving_path, "sample_" + str(self.rank) + "_" + str(ctr))
            ctr += 1

    def eval_time(self, dataset_eval, eval_steps=None, num_evals=10, warmup_eval=True):

        # Evaluation Mode
        self.eval()

        # Warmup Eval
        if warmup_eval:
            print("Warmup Eval")
            self.evaluate(dataset_eval, eval_steps=eval_steps)

        # Eval times
        eval_times = []

        for i in range(num_evals):

            # Print
            print("Eval {}/{}:".format(i+1, num_evals))

            # Start Timer
            start = time.time()

            # Evaluate
            self.evaluate(dataset_eval, eval_steps=eval_steps)

            # Append Eval Time
            eval_times.append(time.time() - start)

        # To Tensor
        eval_times = torch.tensor(eval_times)

        return {"mean": eval_times.mean(), "std": eval_times.std(), "min": eval_times.min(), "max": eval_times.max()}

    def save_logits(self, dataset_eval, callback_path):

        # Evaluation Mode
        self.eval()

        # tqdm Iterator
        if self.rank == 0:
            eval_iterator = tqdm(dataset_eval)
        else: 
            eval_iterator = dataset_eval

        # Logits / Targets List
        logits_list = []
        targets_list = []

        # Loop
        for step, batch in enumerate(eval_iterator):

            # Unpack Batch
            inputs, targets = batch["inputs"], batch["targets"]

            # Transfer Batch elt to model device
            inputs = self.transfer_to_device(inputs)

            # Forward
            with torch.no_grad():
                logits = self.forward(inputs)

            # Transfer Batch elt to cpu device
            logits = self.transfer_to_device(logits, "cpu")

            # Append List
            logits_list.append(logits)
            targets_list.append(targets)

        # Save List
        torch.save(logits_list, os.path.join(callback_path, "logits.pt"))
        torch.save(targets_list, os.path.join(callback_path, "targets.pt"))