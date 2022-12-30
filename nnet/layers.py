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
import torch.nn.functional as F
import torch._VF as _VF
from torch.nn.modules.utils import _single, _pair, _triple

# Initializations
from nnet.initializations import init_dict

###############################################################################
# FC Layers
###############################################################################

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, weight_init="default", bias_init="default"):
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias, device=device, dtype=dtype)

        # Variational Noise
        self.noise = None
        self.vn_std = None

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, x):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise
            
        # Apply Weight
        x = F.linear(x, weight, self.bias)

        return x

###############################################################################
# Convolutional Layers
###############################################################################

class Conv1d(nn.Conv1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=True,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):
        super(Conv1d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=0 if isinstance(padding, str) else padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias, 
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if not isinstance(padding, str):
            padding = "valid"

        # Assert
        assert padding in ["valid", "same", "causal"]

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    (self.kernel_size[0] - 1) // 2, # left
                    self.kernel_size[0] // 2 # right
                ), 
                value=0,
            )

        elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    self.kernel_size[0] // 2, 
                    (self.kernel_size[0] - 1) // 2
                ), 
                value=0
            )

        elif padding == "causal":

            self.pre_padding = nn.ConstantPad1d(
                padding=(
                    self.kernel_size[0] - 1, 
                    0
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class Conv2d(nn.Conv2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        dilation=1, 
        groups=1, 
        bias=True, 
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):
        
        super(Conv2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding if isinstance(padding, int) or isinstance(padding, tuple) else 0, 
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if isinstance(padding, nn.Module):

            self.pre_padding = padding

        elif isinstance(padding, str):

            # Assert
            assert padding in ["valid", "same", "same-left"]

            # Padding
            if padding == "valid":

                self.pre_padding = nn.Identity()

            elif padding == "same":

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        (self.kernel_size[1] - 1) // 2, # left
                        self.kernel_size[1] // 2, # right
                        
                        (self.kernel_size[0] - 1) // 2, # top
                        self.kernel_size[0] // 2 # bottom
                    ), 
                    value=0
                )

            elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

                self.pre_padding = nn.ConstantPad2d(
                    padding=(
                        self.kernel_size[1] // 2,
                        (self.kernel_size[1] - 1) // 2, 

                        self.kernel_size[0] // 2,
                        (self.kernel_size[0] - 1) // 2 
                    ), 
                    value=0
                )
        
        elif isinstance(padding, int) or isinstance(padding, tuple):
            
            self.pre_padding = nn.Identity()

        else:

            raise Exception("Unknown padding: ", padding, type(padding))

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class Conv3d(nn.Conv3d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1, 
        dilation=1, 
        groups=1, 
        bias=True,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        padding="same", 
        channels_last=False,
        weight_init="default",
        bias_init="default",
        mask=None
    ):

        super(Conv3d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding if isinstance(padding, int) or isinstance(padding, tuple) else 0,
            dilation=dilation, 
            groups=groups, 
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        if isinstance(padding, nn.Module):

            self.pre_padding = padding

        elif isinstance(padding, str):   

            # Assert
            assert padding in ["valid", "same", "same-left", "causal", "replicate"]

            # Padding
            if padding == "valid":

                self.pre_padding = nn.Identity()

            elif padding == "same":

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2, # left
                        self.kernel_size[2] // 2, # right
                        
                        (self.kernel_size[1] - 1) // 2, # top
                        self.kernel_size[1] // 2, # bottom
                        
                        (self.kernel_size[0] - 1) // 2, # front
                        self.kernel_size[0] // 2 # back
                    ), 
                    value=0
                )

            elif padding == "replicate":

                self.pre_padding = nn.ReplicationPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2, # left
                        self.kernel_size[2] // 2, # right
                        
                        (self.kernel_size[1] - 1) // 2, # top
                        self.kernel_size[1] // 2, # bottom
                        
                        (self.kernel_size[0] - 1) // 2, # front
                        self.kernel_size[0] // 2 # back
                    )
                )

            elif padding == "same-left": # Prioritize left context rather than right for even kernels, cause strided convolution with even kernel to be asymmetric!

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        self.kernel_size[2] // 2,
                        (self.kernel_size[2] - 1) // 2, 

                        self.kernel_size[1] // 2,
                        (self.kernel_size[1] - 1) // 2, 

                        self.kernel_size[0] // 2,
                        (self.kernel_size[0] - 1) // 2 
                    ), 
                    value=0
                )

            elif padding == "causal":

                self.pre_padding = nn.ConstantPad3d(
                    padding=(
                        (self.kernel_size[2] - 1) // 2,
                        self.kernel_size[2] // 2,
                        
                        (self.kernel_size[1] - 1) // 2,
                        self.kernel_size[1] // 2,
                        
                        self.kernel_size[0] - 1,
                        0
                    ), 
                    value=0
                )

        elif isinstance(padding, int) or isinstance(padding, tuple):
            
            self.pre_padding = nn.Identity()

        else:

            raise Exception("Unknown padding: ", padding, type(padding))

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        # Mask
        if mask == "A":

            stem_kernel_numel = torch.prod(torch.tensor(self.kernel_size))
            mask = torch.cat([
                torch.ones(torch.div(stem_kernel_numel, 2, rounding_mode="floor"), dtype=torch.float32), 
                torch.zeros(torch.div(stem_kernel_numel + 1, 2, rounding_mode="floor"), dtype=torch.float32)
            ], dim=0).reshape(self.kernel_size)

        elif mask == "B":
            
            stem_kernel_numel = torch.prod(torch.tensor(self.kernel_size))
            mask = torch.cat([
                torch.ones(torch.div(stem_kernel_numel + 1, 2, rounding_mode="floor"), dtype=torch.float32), 
                torch.zeros(torch.div(stem_kernel_numel, 2, rounding_mode="floor"), dtype=torch.float32)
            ], dim=0).reshape(self.kernel_size)

        self.register_buffer("mask", mask)

    def forward(self, x):

        # Padding and Permute
        x = self.pre_padding(self.input_permute(x))

        # Mask Filter
        if self.mask != None:
            weight = self.weight * self.mask
        else:
            weight = self.weight

        # Apply Weight
        x = self._conv_forward(x, weight, self.bias)

        # Permute
        x = self.output_permute(x)

        return x

class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        weight_init="default",
        bias_init="default",
        channels_last=False
    ):

        super(ConvTranspose1d, self).__init__(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding,
                output_padding=output_padding, 
                groups=groups, 
                bias=bias,
                dilation=dilation,
                padding_mode=padding_mode,
                device=device,
                dtype=dtype
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def forward(self, x):

        # Apply Weight
        x = self.output_permute(super(ConvTranspose1d, self).forward(self.input_permute(x)))

        return x

class ConvTranspose2d(nn.ConvTranspose2d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        channels_last=False,
        weight_init="default",
        bias_init="default"
    ):

        super(ConvTranspose2d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding, 
            groups=groups, 
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

    def forward(self, x):

        # Apply Weight
        x = self.output_permute(super(ConvTranspose2d, self).forward(self.input_permute(x)))

        return x

class ConvTranspose3d(nn.ConvTranspose3d):

    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride=1,
        padding=0,
        output_padding=0, 
        groups=1, 
        bias=True, 
        dilation=1,
        padding_mode='zeros',
        device=None, 
        dtype=None,

        channels_last=False,
        weight_init="default",
        bias_init="default",
    ):
        super(ConvTranspose3d, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=tuple([k - 1 for k in _triple(kernel_size)]) if isinstance(padding, str) else padding,
            output_padding=output_padding, 
            groups=groups, 
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)
        if bias_init != "default" and self.bias != None:
            if isinstance(bias_init, dict):
                init_dict[bias_init["class"]](self.bias, **bias_init["params"])
            else:
                init_dict[bias_init](self.bias)

        if not isinstance(padding, str):
            padding = "valid"

        # Assert
        assert padding in ["valid", "replicate"]

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "replicate":

            self.pre_padding = nn.ReplicationPad3d(
                padding=(
                    (self.kernel_size[2] - 1) // 2, # left
                    (self.kernel_size[2] - 1) // 2, # right
                    
                    (self.kernel_size[1] - 1) // 2, # top
                    (self.kernel_size[1] - 1) // 2, # bottom
                     
                    (self.kernel_size[0] - 1) // 2, # front
                    (self.kernel_size[0] - 1) // 2 # back
                )
            )

    def forward(self, x):

        return self.output_permute(super(ConvTranspose3d, self).forward(self.input_permute(self.pre_padding(x))))

###############################################################################
# Pooling Layers
###############################################################################

class MaxPool1d(nn.MaxPool1d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool1d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same", "causal"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":
            self.pre_padding = nn.Identity()
        elif padding == "same":
            self.pre_padding = nn.ConstantPad1d(padding=(self.kernel_size[0] // 2, (self.kernel_size[0] - 1) // 2), value=0)
        elif padding == "causal":
            self.pre_padding = nn.ConstantPad1d(padding=(self.kernel_size[0] - 1, 0), value=0)


        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool1d, self).forward(x))

        return x

class MaxPool2d(nn.MaxPool2d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool2d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad2d(
                padding=(
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] // 2,
                    (self.kernel_size[0] - 1) // 2 
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool2d, self).forward(x))

        return x

class MaxPool3d(nn.MaxPool3d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        dilation=1, 
        return_indices=False, 
        ceil_mode=False,

        padding="same",
        channels_last=False
    ):

        super(MaxPool3d, self).__init__(
            kernel_size=_triple(kernel_size),
            stride=stride,
            padding=0,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode
        )

        # Assert
        paddings = ["valid", "same", "causal"]
        assert padding in paddings, "possible paddings are " + ", ".join(paddings)

        # Padding
        if padding == "valid":

            self.pre_padding = nn.Identity()

        elif padding == "same":

            self.pre_padding = nn.ConstantPad3d(
                padding=(
                    self.kernel_size[2] // 2,
                    (self.kernel_size[2] - 1) // 2, 
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] // 2,
                    (self.kernel_size[0] - 1) // 2 
                ), 
                value=0
            )

        elif padding == "causal":

            self.pre_padding = nn.ConstantPad3d(
                padding=(
                    self.kernel_size[2] // 2,
                    (self.kernel_size[2] - 1) // 2, 
                    self.kernel_size[1] // 2,
                    (self.kernel_size[1] - 1) // 2, 
                    self.kernel_size[0] - 1,
                    0
                ), 
                value=0
            )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):

        # Padding
        x = self.pre_padding(self.input_permute(x))

        # Apply Weight
        x = self.output_permute(super(MaxPool3d, self).forward(x))

        return x

class AvgPool1d(nn.AvgPool1d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool1d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=1, to_last=False)
            self.output_permute = PermuteChannels(num_dims=1, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool1d, self).forward(self.input_permute(x)))

class AvgPool2d(nn.AvgPool2d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=2, to_last=False)
            self.output_permute = PermuteChannels(num_dims=2, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool2d, self).forward(self.input_permute(x)))

class AvgPool3d(nn.AvgPool3d):

    def __init__(
        self,
        kernel_size, 
        stride=None, 
        padding=0, 
        ceil_mode=False, 
        count_include_pad=True,
        
        channels_last=False
    ):

        super(AvgPool3d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding, 
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(num_dims=3, to_last=False)
            self.output_permute = PermuteChannels(num_dims=3, to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(AvgPool3d, self).forward(self.input_permute(x)))

class Upsample(nn.Upsample):

    def __init__(
        self,
        size=None, 
        scale_factor=None, 
        mode='nearest', 
        align_corners=None, 
        recompute_scale_factor=None,
        
        channels_last=False
    ):

        super(Upsample, self).__init__(
            size=size,
            scale_factor=scale_factor,
            mode=mode, 
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor
        )

        # Channels Last
        if channels_last:
            self.input_permute = PermuteChannels(to_last=False)
            self.output_permute = PermuteChannels(to_last=True)
        else:
            self.input_permute = nn.Identity()
            self.output_permute = nn.Identity()

    def forward(self, x):
        return self.output_permute(super(Upsample, self).forward(self.input_permute(x)))

###############################################################################
# RNN Layers
###############################################################################

class LSTM(nn.LSTM):

    def __init__(self, input_size, hidden_size, num_layers, batch_first, bidirectional):
        super(LSTM, self).__init__(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=batch_first, 
            bidirectional=bidirectional)

        # Variational Noise
        self.noises = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noises = []
        for i in range(0, len(self._flat_weights), 4):
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i].size(), device=self._flat_weights[i].device, dtype=self._flat_weights[i].dtype))
            self.noises.append(torch.normal(mean=0.0, std=1.0, size=self._flat_weights[i+1].size(), device=self._flat_weights[i+1].device, dtype=self._flat_weights[i+1].dtype))

        # Broadcast Noise
        if distributed:
            for noise in self.noises:
                torch.distributed.broadcast(noise, 0)

    def forward(self, input, hx=None):  # noqa: F811

        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            zeros = torch.zeros(self.num_layers * num_directions,
                                max_batch_size, self.hidden_size,
                                dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        # Add Noise
        if self.noises is not None and self.training:
            weight = []
            for i in range(0, len(self.noises), 2):
                weight.append(self._flat_weights[2*i] + self.vn_std * self.noises[i])
                weight.append(self._flat_weights[2*i+1] + self.vn_std * self.noises[i+1])
                weight.append(self._flat_weights[2*i+2])
                weight.append(self._flat_weights[2*i+3])
        else:
            weight = self._flat_weights

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, weight, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, weight, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_packed = nn.utils.rnn.PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class Embedding(nn.Embedding): 

    def __init__(self, num_embeddings, embedding_dim, padding_idx = None, weight_init="default"):
        super(Embedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx)

        # Init
        if weight_init != "default":
            if isinstance(weight_init, dict):
                init_dict[weight_init["class"]](self.weight, **weight_init["params"])
            else:
                init_dict[weight_init](self.weight)

        # Variational Noise
        self.noise = None
        self.vn_std = None

    def init_vn(self, vn_std):

        # Variational Noise
        self.vn_std = vn_std

    def sample_synaptic_noise(self, distributed):

        # Sample Noise
        self.noise = torch.normal(mean=0.0, std=1.0, size=self.weight.size(), device=self.weight.device, dtype=self.weight.dtype)

        # Broadcast Noise
        if distributed:
            torch.distributed.broadcast(self.noise, 0)

    def forward(self, input):

        # Weight
        weight = self.weight

        # Add Noise
        if self.noise is not None and self.training:
            weight = weight + self.vn_std * self.noise

        # Apply Weight
        return F.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

###############################################################################
# Regularization Layers
###############################################################################

class Dropout(nn.Dropout):

    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__(p, inplace)

    def forward(self, x):

        if self.p > 0:
            return F.dropout(x, self.p, self.training, self.inplace)
        else: 
            return x

###############################################################################
# Tensor Manipulation Layers
###############################################################################

class PermuteChannels(nn.Module):

    """ Permute Channels

    Channels_last to channels_first / channels_first to channels_last
    
    """

    def __init__(self, to_last=True, num_dims=None, make_contiguous=False):
        super(PermuteChannels, self).__init__()

        # To last
        self.to_last = to_last

        # Set dims
        if num_dims != None:
            self.set_dims(num_dims)
        else:
            self.dims = None

        # Make Contiguous
        self.make_contiguous = make_contiguous

    def set_dims(self, num_dims):

        if self.to_last:
            self.dims = (0,) + tuple(range(2, num_dims + 2)) + (1,)
        else:
            self.dims = (0, num_dims + 1) + tuple(range(1, num_dims + 1))

    def forward(self, x):

        if self.dims == None:
            self.set_dims(num_dims=x.dim()-2)

        x = x.permute(self.dims)

        if self.make_contiguous:
            x = x.contiguous()

        return x

class Upsample3d(nn.Upsample):

    def __init__(self, scale_factor):

        # Assert
        if isinstance(scale_factor, int):
            scale_factor = (scale_factor, scale_factor, scale_factor)
        else:
            assert isinstance(scale_factor, list) or isinstance(scale_factor, tuple)
            assert len(scale_factor) == 3

        # Init
        super(Upsample3d, self).__init__(scale_factor=scale_factor)

class Flatten(nn.Flatten):

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)

    def forward(self, x):

        return super(Flatten, self).forward(x)

class Transpose(nn.Module):

    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):

        return x.transpose(self.dim0, self.dim1)

class Permute(nn.Module):

    def __init__(self, dims, make_contiguous=False):
        super(Permute, self).__init__()
        self.dims = dims
        self.make_contiguous = make_contiguous

    def forward(self, x):
        x = x.permute(self.dims)
        if self.make_contiguous:
            x = x.contiguous()
        return x

class Reshape(nn.Module):

    def __init__(self, shape, include_batch=True):
        super(Reshape, self).__init__()
        self.shape = tuple(shape)
        self.include_batch = include_batch

    def forward(self, x):

        if self.include_batch:
            return x.reshape(self.shape)
        else:
            return x.reshape(x.size()[0:1] + self.shape)

class Unsqueeze(nn.Module):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):

        return x.unsqueeze(dim=self.dim)

class GlobalAvgPool1d(nn.Module):

    def __init__(self, dim=1, keepdim=False):
        super(GlobalAvgPool1d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        if mask != None:
            x = (x * mask).sum(dim=self.dim, keepdim=self.keepdim) / mask.count_nonzero(dim=self.dim)
        else:
            x = x.mean(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalAvgPool2d(nn.Module):

    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalAvgPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        if mask != None:
            x = (x * mask).sum(dim=self.dim, keepdim=self.keepdim) / mask.count_nonzero(dim=self.dim)
        else:
            x = x.mean(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalMaxPool2d(nn.Module):

    def __init__(self, dim=(2, 3), keepdim=False):
        super(GlobalMaxPool2d, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x, output_dict=False):

        x = x.amax(dim=self.dim, keepdim=self.keepdim)

        return x

class GlobalAvgPool3d(nn.Module):

    def __init__(self, axis=(2, 3, 4), keepdim=False):
        super(GlobalAvgPool3d, self).__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward(self, x, mask=None):

        return x.mean(axis=self.axis, keepdim=self.keepdim)

###############################################################################
# Layer Dictionary
###############################################################################

layer_dict = {
    "Linear": Linear,

    "Conv1d": Conv1d,
    "Conv2d": Conv2d,
    "Conv3d": Conv3d,

    "ConvTranspose1d": ConvTranspose1d,
    "ConvTranspose2d": ConvTranspose2d,
    "ConvTranspose3d": ConvTranspose3d,

    "MaxPool3d": MaxPool3d,

    "Dropout": Dropout,

    "Flatten": Flatten,
    "Transpose": Transpose,
    "Permute": Permute,
    "Reshape": Reshape,
    "Unsqueeze": Unsqueeze,
    "GlobalAvgPool1d": GlobalAvgPool1d,
    "GlobalAvgPool2d": GlobalAvgPool2d,
    "GlobalAvgPool3d": GlobalAvgPool3d,
    "GlobalMaxPool2d": GlobalMaxPool2d
}