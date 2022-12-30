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
import torch.optim as optim

# NeuralNets
from nnet import embeddings
from nnet import schedulers

###############################################################################
# Optimizers
###############################################################################

class SGD(optim.SGD):

    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(SGD, self).__init__(params=params, lr=torch.tensor(0), momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

        if isinstance(lr, schedulers.Scheduler):
            self.scheduler = lr
        else:
            self.scheduler = schedulers.ConstantScheduler(val=lr)

    def step(self, closure=None):
        lr = self.scheduler.step()
        for group in self.param_groups:
            group['lr'] = lr
        return super(SGD, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(SGD, self).state_dict()

        # Append Scheduler Step
        state_dict["model_step"] = self.scheduler.model_step

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step
        self.scheduler.model_step.fill_(state_dict.pop("model_step"))

        # Load State Dict
        super(SGD, self).load_state_dict(state_dict)

class Adam(optim.Adam):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(Adam, self).__init__(params=params, lr=torch.tensor(0), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        if isinstance(lr, schedulers.Scheduler):
            self.scheduler = lr
        else:
            self.scheduler = schedulers.ConstantScheduler(val=lr)

    def step(self, closure=None):
        lr = self.scheduler.step()
        for group in self.param_groups:
            group['lr'] = lr
        return super(Adam, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(Adam, self).state_dict()

        # Append Scheduler Step
        state_dict["model_step"] = self.scheduler.model_step

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step
        self.scheduler.model_step.fill_(state_dict.pop("model_step"))

        # Load State Dict
        super(Adam, self).load_state_dict(state_dict)

class AdamW(optim.AdamW):

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False):
        super(AdamW, self).__init__(params=params, lr=torch.tensor(0), betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        if isinstance(lr, schedulers.Scheduler):
            self.scheduler = lr
        else:
            self.scheduler = schedulers.ConstantScheduler(val=lr)

    def step(self, closure=None):
        lr = self.scheduler.step()
        for group in self.param_groups:
            group['lr'] = lr
        return super(AdamW, self).step(closure)

    def state_dict(self):

        # Get State Dict
        state_dict = super(AdamW, self).state_dict()

        # Append Scheduler Step
        state_dict["model_step"] = self.scheduler.model_step

        return state_dict

    def load_state_dict(self, state_dict):

        # Load Scheduler Step
        self.scheduler.model_step.fill_(state_dict.pop("model_step"))

        # Load State Dict
        super(AdamW, self).load_state_dict(state_dict)

###############################################################################
# Utils
###############################################################################

def get_decay_param_groups(model, weight_decay=0.01, decay_modules=(torch.nn.Linear,), no_decay_modules=(torch.nn.LayerNorm, torch.nn.Embedding, embeddings.PosEmbedding1d), decay_params=("weight",), no_decay_params=("bias",)):

    # Init Decay / No decay Sets
    decay = set()
    no_decay = set()

    # Param Loop
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters():

            # Full Param Name
            full_param_name = "{}.{}".format(module_name, param_name) if module_name else param_name

            # No Decay params
            for no_decay_param in no_decay_params:
                if param_name.endswith(no_decay_param):
                    no_decay.add(full_param_name)
                    continue

            # Decay params
            for decay_param in decay_params:

                # Decay Weight
                if param_name.endswith(decay_param) and isinstance(module, decay_modules):
                    decay.add(full_param_name)
                    continue

                # No Decay Weight
                if param_name.endswith(decay_param) and isinstance(module, no_decay_modules):
                    no_decay.add(full_param_name)
                    continue

    # Validate that we considered every parameter
    param_dict = {param_name: param for param_name, param in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters {} made it into both decay/no_decay sets!".format(str(inter_params))
    assert len(param_dict.keys() - union_params) == 0, "parameters {} were not separated into either decay/no_decay set!".format(str(param_dict.keys() - union_params))

    # Create Param Groups
    param_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    return param_groups

###############################################################################
# Optimizer Dictionary
###############################################################################

optim_dict = {
    "SGD": SGD,
    "RMSprop": optim.RMSprop,
    "Adam": Adam,
    "AdamW": AdamW
}