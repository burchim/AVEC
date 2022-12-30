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
from nnet import activations
from nnet import normalizations
from nnet import attentions

###############################################################################
# Modules
###############################################################################

class MultiLayerPerceptron(nn.Module):

    def __init__(self, dim_input, dim_layers, act_fun="ReLU", norm=None, drop_rate=0.0):
        super(MultiLayerPerceptron, self).__init__()

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = activations.act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = activations.act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = normalizations.norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = normalizations.norm_dict[norm]

        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]
            
        # MLP Layers
        self.layers = nn.ModuleList([nn.Sequential(
            layers.Linear(dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id]),
            norm(dim_layers[layer_id], **norm_params),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x):

        # Layers
        for layer in self.layers:
            x = layer(x)

        return x

class ConvNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, strides=1, norm=None, act_fun="ReLU", drop_rate=0.0, padding="same", dim=2, channels_last=False, residual=False, weight_init="default", bias_init="default", bias=True):
        super(ConvNeuralNetwork, self).__init__()

        conv = {
            1: layers.Conv1d,
            2: layers.Conv2d,
            3: layers.Conv3d
        }

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = activations.act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = activations.act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = normalizations.norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = normalizations.norm_dict[norm]

        # Strides
        self.strides = strides

        # Residual
        self.residual = residual

        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]

        # CNN Layers
        self.layers = nn.ModuleList([nn.Sequential(
            conv[dim](dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id], kernel_size[layer_id] if isinstance(kernel_size, list) else kernel_size, stride=strides[layer_id] if isinstance(strides, list) else strides, padding=padding[layer_id] if isinstance(padding, list) else padding, channels_last=channels_last, weight_init=weight_init, bias_init=bias_init, bias=bias), 
            norm(dim_layers[layer_id], **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x, x_len=None):

        # Layers
        for layer in self.layers:

            # Forward
            if self.residual:
                x = x + layer(x)
            else:
                x = layer(x)

            # Update Sequence Lengths
            if x_len is not None:
                x_len = torch.div(x_len - 1, 2, rounding_mode='floor') + 1 # to generalize

        return x if x_len==None else (x, x_len)

class ConvTransposeNeuralNetwork(nn.Module):

    def __init__(self, dim_input, dim_layers, kernel_size, padding=0, output_padding=0, strides=1, norm=None, act_fun="ReLU", drop_rate=0.0, dim=2, channels_last=False, weight_init="default", bias_init="default", bias=True):
        super(ConvTransposeNeuralNetwork, self).__init__()

        conv = {
            1: layers.ConvTranspose1d,
            2: layers.ConvTranspose2d,
            3: layers.ConvTranspose3d
        }

        # Act fun
        if isinstance(act_fun, dict):
            act_fun_params = act_fun["params"]
            act_fun = activations.act_dict[act_fun["class"]]
        else:
            act_fun_params = {}
            act_fun = activations.act_dict[act_fun]

        # Norm
        if isinstance(norm, dict):
            norm_params = norm["params"]
            norm = normalizations.norm_dict[norm["class"]]
        else:
            norm_params = {}
            norm = normalizations.norm_dict[norm]

        # Strides
        self.strides = strides

        # Single Layer
        if isinstance(dim_layers, int):
            dim_layers = [dim_layers]

        # CNN Layers
        self.layers = nn.ModuleList([nn.Sequential(
            conv[dim](dim_input if layer_id == 0 else dim_layers[layer_id - 1], dim_layers[layer_id], kernel_size[layer_id] if isinstance(kernel_size, list) else kernel_size, stride=strides[layer_id] if isinstance(strides, list) else strides, padding=padding[layer_id] if isinstance(padding, list) else padding, output_padding=output_padding[layer_id] if isinstance(output_padding, list) else output_padding, channels_last=channels_last, weight_init=weight_init, bias_init=bias_init, bias=bias), 
            norm(dim_layers[layer_id], **norm_params, channels_last=channels_last),
            act_fun(**act_fun_params),
            nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity() 
        ) for layer_id in range(len(dim_layers))])

    def forward(self, x, x_len=None):

        # Layers
        for layer in self.layers:

            x = layer(x)

        return x if x_len==None else (x, x_len)

###############################################################################
# Residual CNN Modules
###############################################################################

class InceptionModule(nn.Module):

    """ GoogLeNet Inception Module

    References: "Going deeper with convolutions", Szegedy et al.
    https://arxiv.org/abs/1409.4842

    args:
        in_channels: number of input channels
        out_channels: list of branches output channels [C0, C1, C2, C3, C4, C5]
        kernel_sizes: branch 1 and 2 kernel sizes [K0, K1]
        dim: Module dimension

    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], dim=2, inplace=False):
        super(InceptionModule, self).__init__()

        conv = layers.layer_dict["Conv" + str(dim) + "d"]
        norm = normalizations.norm_dict["BatchNorm" + str(dim) + "d"]
        pool = layers.layer_dict["MaxPool" + str(dim) + "d"]

        # Branch 0
        self.branch_0 = nn.Sequential(
            conv(in_channels, out_channels[0], kernel_size=1, bias=False),
            norm(out_channels[0]),
            nn.ReLU(inplace=inplace)
        )

        # Branch 1
        self.branch_1 = nn.Sequential(
            conv(in_channels, out_channels[1], kernel_size=1, bias=False),
            norm(out_channels[1]),
            nn.ReLU(inplace=inplace),
            conv(out_channels[1], out_channels[2], kernel_size=kernel_sizes[0], bias=False),
            norm(out_channels[2]),
            nn.ReLU(inplace=inplace)
        )

        # Branch 2
        self.branch_2 = nn.Sequential(
            conv(in_channels, out_channels[3], kernel_size=1, bias=False),
            norm(out_channels[3]),
            nn.ReLU(inplace=inplace),
            conv(out_channels[3], out_channels[4], kernel_size=kernel_sizes[1], bias=False),
            norm(out_channels[4]),
            nn.ReLU(inplace=inplace)
        )

        # Branch 3
        self.branch_3 = nn.Sequential(
            pool(kernel_size=3, stride=1),
            conv(in_channels, out_channels[5], kernel_size=1, bias=False),
            norm(out_channels[5]),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):

        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)

        return torch.cat([x_0, x_1, x_2, x_3], dim=1)

###############################################################################
# Transformer Modules
###############################################################################

class FeedForwardModule(nn.Module):

    """Transformer Feed Forward Module

    Args:
        dim_model: model feature dimension
        dim_ffn: expanded feature dimension
        Pdrop: dropout probability
        act: inner activation function
        inner_dropout: whether to apply dropout after the inner activation function

    Input: (batch size, length, dim_model)
    Output: (batch size, length, dim_model)
    
    """

    def __init__(self, dim_model, dim_ffn, drop_rate, act_fun, inner_dropout, prenorm=True, weight_init="default", bias_init="default"):
        super(FeedForwardModule, self).__init__()

        # Layers
        self.layers = nn.Sequential(
            nn.LayerNorm(dim_model, eps=1e-6) if prenorm else nn.Identity(),
            layers.Linear(dim_model, dim_ffn, weight_init=weight_init, bias_init=bias_init),
            activations.act_dict[act_fun](),
            nn.Dropout(p=drop_rate) if inner_dropout else nn.Identity(),
            layers.Linear(dim_ffn, dim_model, weight_init=weight_init, bias_init=bias_init),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):

        # Layers
        return self.layers(x)

class AttentionModule(nn.Module):

    """ Attention Module

    Args:
        dim_model: model feature dimension
        att_params: attention params
        drop_rate: residual dropout probability

    """

    def __init__(self, dim_model, att_params, drop_rate, norm={"class": "LayerNorm", "params": {"eps": 1e-6}}, residual=True, channels_last=True):
        super(AttentionModule, self).__init__()

        # Pre Norm
        if isinstance(norm, dict):
            self.norm = normalizations.norm_dict[norm["class"]](dim_model, **norm["params"], channels_last=channels_last)
        else:
            self.norm = normalizations.norm_dict[norm](dim_model, channels_last=channels_last)

        # Attention
        self.attention = attentions.att_dict[att_params["class"]](dim_model=dim_model, **att_params["params"])
            
        # Dropout
        self.dropout = nn.Dropout(drop_rate)

        # Params
        self.residual = residual

    def forward(self, x, x_cross=None, mask=None):

        # Residual
        if self.residual:
            x_res = x

        # Pre Norm
        x = self.norm(x)

        # Self-Attention
        x = self.attention.forwardQKV(Q=x, K=x_cross if x_cross != None else x, V=x_cross if x_cross != None else x, mask=mask)

        # Dropout
        x = self.dropout(x)

        # Residual
        if self.residual:
            x = x + x_res

        return x

class ConvolutionModule(nn.Module):

    """Conformer Convolution Module

    Args:
        dim_model: input feature dimension
        dim_expand: output feature dimension
        kernel_size: depthwise convolution kernel size
        drop_rate: residual dropout probability
        stride: depthwise convolution stride
        padding: "valid", "same" or "causal"
        dim: number of spatiotemporal input dimensions
        channels_last: ordering of the dimensions in the inputs

    References: 
        https://arxiv.org/abs/2005.08100
    
    """

    def __init__(self, dim_model, dim_expand, drop_rate, stride, act_fun="Swish", conv_params={"class": "Conv2d", "params":{"padding":"same", "kernel_size": 3}}, channels_last=False, batch_norm=True):
        super(ConvolutionModule, self).__init__()

        # Layers
        pointwise_conv = layers.layer_dict[conv_params["class"].replace("Transpose", "")]
        depthwise_conv = layers.layer_dict[conv_params["class"]]
        if batch_norm:
            norm = normalizations.norm_dict[conv_params["class"].replace("Transpose", "").replace("Conv", "BatchNorm")]
        else:
            norm = normalizations.LayerNorm

        # Layers
        self.layers = nn.Sequential(
            normalizations.LayerNorm(dim_model, channels_last=channels_last, eps=1e-6),
            pointwise_conv(dim_model, 2 * dim_expand, kernel_size=1, channels_last=channels_last),
            nn.GLU(dim=-1 if channels_last else 1),
            depthwise_conv(dim_expand, dim_expand, stride=stride, groups=dim_expand, channels_last=channels_last, **conv_params["params"]),
            norm(dim_expand, channels_last=channels_last),
            activations.act_dict[act_fun](),
            pointwise_conv(dim_expand, dim_expand, kernel_size=1, channels_last=channels_last),
            nn.Dropout(p=drop_rate)
        )

    def forward(self, x):

        return self.layers(x)

class InterCTCResModule(nn.Module):

    def __init__(self, dim_model, vocab_size):
        super(InterCTCResModule, self).__init__()

        self.proj_1 = layers.Linear(dim_model, vocab_size)
        self.proj_2 = layers.Linear(vocab_size, dim_model)

    def forward(self, x):

        logits = self.proj_1(x)
        x = x + self.proj_2(logits.softmax(dim=-1))

        return x, logits

class FusionModule(nn.Module):

    def __init__(self, a_dim_model=360, v_dim_model=360, f_dim_model=360, ff_ratio=4):
        super(FusionModule, self).__init__()

        dim_in = a_dim_model + v_dim_model
        dim_ffn = ff_ratio * f_dim_model
        dim_out = f_dim_model
        weight_init = "default"
        bias_init = "default"
        act_fun = "Swish"

        # Layers
        self.layers = nn.Sequential(
            layers.Linear(dim_in, dim_ffn, weight_init=weight_init, bias_init=bias_init),
            activations.act_dict[act_fun](),
            layers.Linear(dim_ffn, dim_out, weight_init=weight_init, bias_init=bias_init),
        )

    def forward(self, audio, video):

        x = torch.cat([audio, video], dim=-1)
        x = self.layers(x)

        return x