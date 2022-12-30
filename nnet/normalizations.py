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

# NeuralNets
from nnet import layers
from nnet import initializations

###############################################################################
# Normalization Layers
###############################################################################

class LayerNorm(nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None, channels_last=True):
        super(LayerNorm, self).__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.transpose = nn.Identity()
        else:
            self.transpose = layers.Transpose(dim0=1, dim1=-1)

    def forward(self, input):

        return self.transpose(super(LayerNorm, self).forward(self.transpose(input)))

class BatchNorm1d(nn.BatchNorm1d):

    """
    
    args:
        frozen: eval mode is used for both training and inference
    
    """

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False, weight_init="default", bias_init="default", frozen=False):
        super(BatchNorm1d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Frozen Mode
        self.frozen = frozen

        # Channels Last
        if channels_last:
            self.input_permute = layers.Permute(dims=(0, 2, 1))
            self.output_permute = layers.Permute(dims=(0, 2, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default" and self.affine:
            if isinstance(weight_init, dict):
                initializations.init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                initializations.init_dict[weight_init](self.weight)
        if bias_init != "default" and self.affine:
            if isinstance(bias_init, dict):
                initializations.init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                initializations.init_dict[bias_init](self.bias)

    def forward(self, input):

        if self.frozen:
            mode = self.training
            self.eval()

        output = self.output_permute(super(BatchNorm1d, self).forward(self.input_permute(input)))

        if self.frozen:
            self.train(mode)

        return output

class BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False, weight_init="default", bias_init="default", frozen=False):
        super(BatchNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Frozen Mode
        self.frozen = frozen

        # Channels Last
        if channels_last:
            self.input_permute = layers.Permute(dims=(0, 3, 1, 2))
            self.output_permute = layers.Permute(dims=(0, 2, 3, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default" and self.affine:
            if isinstance(weight_init, dict):
                initializations.init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                initializations.init_dict[weight_init](self.weight)
        if bias_init != "default" and self.affine:
            if isinstance(bias_init, dict):
                initializations.init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                initializations.init_dict[bias_init](self.bias)

    def forward(self, input):

        if self.frozen:
            mode = self.training
            self.eval()

        output = self.output_permute(super(BatchNorm2d, self).forward(self.input_permute(input)))

        if self.frozen:
            self.train(mode)

        return output

class BatchNorm3d(nn.BatchNorm3d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None, channels_last=False, weight_init="default", bias_init="default", frozen=False):
        super(BatchNorm3d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Frozen Mode
        self.frozen = frozen

        # Channels Last
        if channels_last:
            self.input_permute = layers.Permute(dims=(0, 4, 1, 2, 3))
            self.output_permute = layers.Permute(dims=(0, 2, 3, 4, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default" and self.affine:
            if isinstance(weight_init, dict):
                initializations.init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                initializations.init_dict[weight_init](self.weight)
        if bias_init != "default" and self.affine:
            if isinstance(bias_init, dict):
                initializations.init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                initializations.init_dict[bias_init](self.bias)

    def forward(self, input):

        if self.frozen:
            mode = self.training
            self.eval()

        output = self.output_permute(super(BatchNorm3d, self).forward(self.input_permute(input)))

        if self.frozen:
            self.train(mode)

        return output

class SyncBatchNorm(nn.SyncBatchNorm):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, process_group=None, device=None, dtype=None, channels_last=False, weight_init="default", bias_init="default", frozen=False):
        super(SyncBatchNorm, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Frozen Mode
        self.frozen = frozen

        # Channels Last
        if channels_last:
            self.input_permute = layers.PermuteChannels(to_last=False)
            self.output_permute = layers.PermuteChannels(to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default" and self.affine:
            if isinstance(weight_init, dict):
                initializations.init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                initializations.init_dict[weight_init](self.weight)
        if bias_init != "default" and self.affine:
            if isinstance(bias_init, dict):
                initializations.init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                initializations.init_dict[bias_init](self.bias)

    def forward(self, input):

        if self.frozen:
            mode = self.training
            self.eval()

        output = self.output_permute(super(SyncBatchNorm, self).forward(self.input_permute(input)))

        if self.frozen:
            self.train(mode)

        return output

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):

        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked

            # Redefinition attributes
            if hasattr(module, "input_permute"):
                module_output.input_permute = module.input_permute
            if hasattr(module, "output_permute"):
                module_output.output_permute = module.output_permute
            if hasattr(module, "frozen"):
                module_output.frozen = module.frozen

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output

class InstanceNorm2d(nn.InstanceNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, device=None, dtype=None, channels_last=False):
        super(InstanceNorm2d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = layers.Permute(dims=(0, 3, 1, 2))
            self.output_permute = layers.Permute(dims=(0, 2, 3, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(InstanceNorm2d, self).forward(self.input_permute(input)))

class InstanceNorm3d(nn.InstanceNorm3d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False, device=None, dtype=None, channels_last=False):
        super(InstanceNorm3d, self).__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = layers.Permute(dims=(0, 4, 1, 2, 3))
            self.output_permute = layers.Permute(dims=(0, 2, 3, 4, 1))
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(InstanceNorm3d, self).forward(self.input_permute(input)))

class GroupNorm(nn.GroupNorm):

    def __init__(self, num_features, num_groups, eps=1e-05, affine=True, device=None, dtype=None, channels_last=False):
        super(GroupNorm, self).__init__(num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine, device=device, dtype=dtype)

        # Channels Last
        if channels_last:
            self.input_permute = layers.PermuteChannels(to_last=False)
            self.output_permute = layers.PermuteChannels(to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, input):

        return self.output_permute(super(GroupNorm, self).forward(self.input_permute(input).contiguous()))

###############################################################################
# Normalization Dictionary
###############################################################################

norm_dict = {
    None: nn.Identity,
    "LayerNorm": LayerNorm,
    "BatchNorm1d": BatchNorm1d,
    "BatchNorm2d": BatchNorm2d,
    "BatchNorm3d": BatchNorm3d,
    "InstanceNorm2d": InstanceNorm2d,
    "InstanceNorm3d": InstanceNorm3d,
    "LayerNorm": LayerNorm,
    "GroupNorm": GroupNorm
}