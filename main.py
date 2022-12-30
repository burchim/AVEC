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

# Functions
import functions

# Other
import os
import argparse
import importlib
import warnings

# Disable Warnings
warnings.filterwarnings("ignore")

def main(rank, args):

    ###############################################################################
    # Init
    ###############################################################################

    # Process rank
    args.rank = rank

    # Print Mode
    if args.rank == 0:
        print("Mode: {}".format(args.mode))

    # Distributed Computing
    if args.distributed:
        torch.cuda.set_device(args.rank)
        torch.distributed.init_process_group(backend=args.backend, init_method='env://', world_size=args.world_size, rank=args.rank)

    # Load Config
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))

    # Load Model
    model = functions.load_model(args)

    # Load Dataset
    dataset_train, dataset_eval = functions.load_datasets(args)

    ###############################################################################
    # Modes
    ###############################################################################

    assert args.mode in ["training", "evaluation", "swa", "pass", "eval_time"]

    # Training
    if args.mode == "training":

        model.fit(
            dataset_train=dataset_train, 
            epochs=getattr(args.config, "epochs", 1000), 
            dataset_eval=dataset_eval, 
            eval_steps=getattr(args.config, "eval_steps", args.eval_steps), 
            verbose_eval=args.verbose_eval, 
            initial_epoch=int(args.checkpoint.split("_")[2]) if args.checkpoint != None else 0, 
            callback_path=args.config.callback_path, 
            steps_per_epoch=args.steps_per_epoch,
            precision=getattr(args.config, "precision", torch.float32),
            accumulated_steps=getattr(args.config, "accumulated_steps", 1),
            eval_period_step=getattr(args.config, "eval_period_step", args.eval_period_step),
            eval_period_epoch=getattr(args.config, "eval_period_epoch", args.eval_period_epoch),
            saving_period_step=getattr(args.config, "saving_period_step", args.saving_period_step),
            saving_period_epoch=getattr(args.config, "saving_period_epoch", args.saving_period_epoch),
            log_figure_period_step=getattr(args.config, "log_figure_period_step", args.log_figure_period_step),
            log_figure_period_epoch=getattr(args.config, "log_figure_period_epoch", args.log_figure_period_epoch),
            step_log_period=args.step_log_period,
            eval_training=getattr(args.config, "eval_training", not args.no_eval_training),
            dist_log=args.dist_log,
            grad_init_scale=getattr(args.config, "grad_init_scale", 65536.0),
            detect_anomaly=getattr(args.config, "detect_anomaly", args.detect_anomaly),
            recompute_metrics=getattr(args.config, "recompute_metrics", False)
        )

    # Evaluation
    elif args.mode == "evaluation":

        model._evaluate(
            dataset_eval, 
            writer=None, 
            step=None, 
            eval_steps=getattr(args.config, "eval_steps", args.eval_steps), 
            verbose=args.verbose_eval, 
            recompute_metrics=getattr(args.config, "recompute_metrics", False)
        )

    # Stochastic Weight Averaging
    elif args.mode == "swa":

        model.swa(dataset_train, callback_path=args.config.callback_path, start_epoch=args.swa_epochs[0] if args.swa_epochs else None, end_epoch=args.swa_epochs[1] if args.swa_epochs else None, epochs_list=args.swa_epochs_list, update_steps=args.steps_per_epoch, swa_type=args.swa_type, precision=args.config.precision)

    # Pass
    elif args.mode == "pass":
        pass

    # Eval Time
    elif args.mode == "eval_time":
        eval_time = model.eval_time(dataset_eval, eval_steps=getattr(args.config, "eval_steps", args.eval_steps))
        if args.rank == 0:
            print("Eval time: {}".format(eval_time))
            
    ###############################################################################
    # Clean
    ###############################################################################

    # Destroy Process Group
    if args.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file",          type=str,   default="configs/LRS23/AV/EffConfInterCTC.py",      help="Python configuration file containing model hyperparameters")
    parser.add_argument("-m", "--mode",                 type=str,   default="training",                                 help="Mode : training, evaluation, swa, pass, eval_time")
    parser.add_argument("-i", "--checkpoint",           type=str,   default=None,                                       help="Load model from checkpoint name")
    parser.add_argument("-j", "--num_workers",          type=int,   default=0,                                          help="Number of data loading workers")
    parser.add_argument("--cpu",                        action="store_true",                                            help="Load model on cpu")
    parser.add_argument("--load_last",                  action="store_true",                                            help="Load last model checkpoint")

    # Distributed
    parser.add_argument("-d", "--distributed",          action="store_true",                                            help="Distributed data parallelization")
    parser.add_argument("--parallel",                   action="store_true",                                            help="Parallelize model using data parallelization")
    parser.add_argument("--world_size",                 type=int,   default=torch.cuda.device_count(),                  help="Number of available GPUs")
    parser.add_argument("--dist_log",                   action="store_true",                                            help="Log each GPU process instead only GPU:0")
    parser.add_argument("--dist_addr",                  type=str,   default='localhost',                                help="MASTER_ADDR")
    parser.add_argument("--dist_port",                  type=str,   default='29501',                                    help="MASTER_PORT")
    parser.add_argument("--backend",                    type=str,   default='nccl',                                     help="backend")

    # Training
    parser.add_argument("--steps_per_epoch",            type=int,   default=None,                                       help="Number of steps per epoch")
    parser.add_argument("--saving_period_step",         type=int,   default=None,                                       help="Model saving every 'n' steps")
    parser.add_argument("--saving_period_epoch",        type=int,   default=1,                                          help="Model saving every 'n' epochs")
    parser.add_argument("--log_figure_period_step",     type=int,   default=None,                                       help="Log figure every 'n' steps")
    parser.add_argument("--log_figure_period_epoch",    type=int,   default=1,                                          help="Log figure every 'n' epochs")
    parser.add_argument("--step_log_period",            type=int,   default=100,                                        help="Training step log period")
    parser.add_argument("--no_eval_training",           action="store_true",                                            help="Do not evaluate training samples")

    # Eval
    parser.add_argument("--eval_period_epoch",          type=int,   default=1,                                          help="Model evaluation every 'n' epochs")
    parser.add_argument("--eval_period_step",           type=int,   default=None,                                       help="Model evaluation every 'n' steps")
    parser.add_argument("--batch_size_eval",            type=int,   default=None,                                       help="Evaluation batch size")
    parser.add_argument("--verbose_eval",               type=int,   default=0,                                          help="Evaluation verbose level")
    parser.add_argument("--eval_steps",                 type=int,   default=None,                                       help="Number of evaluation steps")

    # Info
    parser.add_argument("--show_dict",                  action="store_true",                                            help="Show model dict summary")
    parser.add_argument("--show_modules",               action="store_true",                                            help="Show model named modules")
    
    # SWA
    parser.add_argument("--swa_epochs",                 nargs="+",  default=None,                                       help="Start epoch / end epoch for swa")
    parser.add_argument("--swa_epochs_list",            nargs="+",  default=None,                                       help="List of checkpoints epochs for swa")
    parser.add_argument("--swa_type",                   type=str,   default="equal",                                    help="Stochastic weight averaging type (equal/exp)")

    # Debug
    parser.add_argument("--detect_anomaly",             action="store_true",                                            help="Enable or disable the autograd anomaly detection")
    
    # Parse Args
    args = parser.parse_args()

    # Run main
    if args.distributed:
        args.dist_port = str(functions.get_open_port())
        os.environ['MASTER_ADDR'] = args.dist_addr
        os.environ['MASTER_PORT'] = args.dist_port
        print("Distributed Mode")
        print("MASTER_ADDR: {}".format(args.dist_addr))
        print("MASTER_PORT: {}".format(args.dist_port))
        print("world_size: {}".format(args.world_size))
        print("backend: {}".format(args.backend))
        torch.multiprocessing.spawn(main, nprocs=args.world_size, args=(args,))  
    else:
        main(0, args)
