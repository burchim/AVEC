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
from nnet import preprocessing
from nnet import layers
from nnet import modules
from nnet import blocks
from nnet import attentions
from nnet import transforms
from nnet import normalizations

###############################################################################
# Networks
###############################################################################

class ResNet(nn.Module):

    """ ResNet (ResNet18, ResNet34, ResNet50, ResNet101, ResNet152)

    Models: 224 x 224
    ResNet18: 11,689,512 Params
    ResNet34: 21,797,672 Params
    ResNet50: 25,557,032 Params
    ResNet101: 44,549,160 Params
    Resnet152: 60,192,808 Params

    Reference: "Deep Residual Learning for Image Recognition" by He et al.
    https://arxiv.org/abs/1512.03385

    """

    def __init__(self, dim_input=3, dim_output=1000, model="ResNet50", include_stem=True, include_head=True):
        super(ResNet, self).__init__()

        assert model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]

        if model == "ResNet18":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [2, 2, 2, 2]
            bottleneck = False
        elif model == "ResNet34":
            dim_stem = 64
            dim_blocks = [64, 128, 256, 512]
            num_blocks = [3, 4, 6, 3]
            bottleneck = False
        elif model == "ResNet50":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            bottleneck = True
        elif model == "ResNet101":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 23, 3]
            bottleneck = True
        elif model == "ResNet152":
            dim_stem = 64
            dim_blocks = [256, 512, 1024, 2048]
            num_blocks = [3, 8, 36, 3]
            bottleneck = True

        self.stem = nn.Sequential(
            layers.Conv2d(in_channels=dim_input, out_channels=dim_stem, kernel_size=(7, 7), stride=(2, 2), weight_init="he_normal", bias=False),
            normalizations.BatchNorm2d(num_features=dim_stem),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        ) if include_stem else nn.Identity()

        # Blocks
        self.blocks = nn.ModuleList()
        for stage_id in range(4):

            for block_id in range(num_blocks[stage_id]):

                # Projection Block
                if block_id == 0:
                    if stage_id == 0:
                        stride = (1, 1)
                        bottleneck_ratio = 1
                        in_features = dim_stem
                    else:
                        stride = (2, 2)
                        bottleneck_ratio = 2
                        in_features = dim_blocks[stage_id-1]
                # Default Block
                else:
                    stride = (1, 1)
                    in_features = dim_blocks[stage_id]
                    bottleneck_ratio = 4

                if bottleneck:
                    self.blocks.append(blocks.ResNetBottleneckBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        bottleneck_ratio=bottleneck_ratio,
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))
                else:
                    self.blocks.append(blocks.ResNetBlock(
                        in_features=in_features,
                        out_features=dim_blocks[stage_id],
                        kernel_size=(3, 3),
                        stride=stride,
                        act_fun="ReLU",
                        joined_post_act=True
                    ))

        # Head
        self.head = nn.Sequential(
            layers.GlobalAvgPool2d(),
            layers.Linear(in_features=dim_blocks[-1], out_features=dim_output, weight_init="he_normal", bias_init="zeros")
        ) if include_head else nn.Identity()

    def forward(self, x):

        # (B, Din, H, W) -> (B, D0, H//4, W//4)
        x = self.stem(x)

        # (B, D0, H//4, W//4) -> (B, D4, H//32, W//32)
        for block in self.blocks:
            x = block(x)

        # (B, D4, H//32, W//32) -> (B, Dout)
        x = self.head(x)

        return x

class Transformer(nn.Module):

    def __init__(self, dim_model, num_blocks, att_params={"class": "MultiHeadAttention", "params":{"num_heads": 4, "weight_init": "normal_02", "bias_init": "zeros"}}, ff_ratio=4, emb_drop_rate=0.1, drop_rate=0.1, act_fun="GELU", pos_embedding=None, mask=None, inner_dropout=False, weight_init="normal_02", bias_init="zeros", post_norm=False):
        super(Transformer, self).__init__()

        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=emb_drop_rate)

        # Mask
        self.mask = mask

        # Transformer Blocks
        self.blocks = nn.ModuleList([blocks.TransformerBlock(
            dim_model=dim_model,
            ff_ratio=ff_ratio,
            att_params=att_params,
            drop_rate=drop_rate,
            inner_dropout=inner_dropout,
            act_fun=act_fun,
            weight_init=weight_init,
            bias_init=bias_init,
            post_norm=post_norm
        ) for block_id in range(num_blocks)])

        # LayerNorm
        self.layernorm = nn.LayerNorm(normalized_shape=dim_model) if not post_norm else nn.Identity()

    def forward(self, x, lengths=None):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)

        # Input Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask=mask)

        # LayerNorm
        x = self.layernorm(x)

        return x

class ConformerInterCTC(nn.Module):

    def __init__(self, dim_model, num_blocks, interctc_blocks, vocab_size, loss_prefix="ctc", att_params={"class": "MultiHeadAttention", "num_heads": 4}, conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": 31}}, ff_ratio=4, drop_rate=0.1, pos_embedding=None, mask=None, conv_stride=1, batch_norm=True):
        super(ConformerInterCTC, self).__init__()

        # Inter CTC Params
        self.interctc_blocks = interctc_blocks
        self.loss_prefix = loss_prefix

        # Single Stage
        if isinstance(dim_model, int):
            dim_model = [dim_model]
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks]
        
        # Positional Embedding
        self.pos_embedding = pos_embedding

        # Input Dropout
        self.dropout = nn.Dropout(p=drop_rate)

        # Mask
        self.mask = mask

        # Conformer Stages
        i = 1
        self.conformer_blocks = nn.ModuleList()
        self.interctc_modules = nn.ModuleList()
        for stage_id in range(len(num_blocks)):

            # Conformer Blocks
            for block_id in range(num_blocks[stage_id]):

                # Transposed Block
                transposed_block = "Transpose" in conv_params["class"]

                # Downsampling Block
                down_block = ((block_id == 0) and (stage_id > 0)) if transposed_block else ((block_id == num_blocks[stage_id] - 1) and (stage_id < len(num_blocks) - 1))

                # Block
                self.conformer_blocks.append(blocks.ConformerBlock(
                    dim_model=dim_model[stage_id - (1 if transposed_block and down_block else 0)],
                    dim_expand=dim_model[stage_id + (1 if not transposed_block and down_block else 0)],
                    ff_ratio=ff_ratio,
                    drop_rate=drop_rate,
                    att_params=att_params[stage_id - (1 if transposed_block and down_block else 0)] if isinstance(att_params, list) else att_params,
                    conv_stride=1 if not down_block else conv_stride[stage_id] if isinstance(conv_stride, list) else conv_stride,
                    conv_params=conv_params[stage_id] if isinstance(conv_params, list) else conv_params,
                    batch_norm=batch_norm
                ))

                # InterCTC Block
                if i in interctc_blocks:
                    self.interctc_modules.append(modules.InterCTCResModule(
                        dim_model=dim_model[stage_id + (1 if not transposed_block and down_block else 0)], 
                        vocab_size=vocab_size
                    ))

                i += 1

    def forward(self, x, lengths):

        # Pos Embedding
        if self.pos_embedding != None:
            x = self.pos_embedding(x)
            
        # Dropout
        x = self.dropout(x)

        # Mask (1 or B, 1, N, N)
        if self.mask != None:
            mask = self.mask(x, lengths)
        else:
            mask = None

        # Conformer Blocks
        interctc_outputs = {}
        j = 0
        for i, block in enumerate(self.conformer_blocks):

            # Conformer Block
            x = block(x, mask=mask)

            # InterCTC Block
            if i + 1 in self.interctc_blocks:
                x, logits = self.interctc_modules[j](x)
                j += 1
                key = self.loss_prefix + "_" + str(i)
            else:
                logits = None

            # Strided Block
            if block.stride > 1:

                # Stride Mask (1 or B, 1, T // S, T // S)
                if mask is not None:
                    mask = mask[:, :, ::block.stride, ::block.stride]

                # Update Seq Lengths
                if lengths is not None:
                    lengths = torch.div(lengths - 1, block.stride, rounding_mode='floor') + 1

            if logits != None:
                interctc_outputs[key] = [logits, lengths]

        return x, lengths, interctc_outputs

class AudioEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, att_type="patch", interctc_blocks=[3, 6, 10, 13], num_blocks=[5, 6, 5], loss_prefix="ctc"):
        super(AudioEfficientConformerEncoder, self).__init__()

        assert att_type in ["regular", "grouped", "patch"]

        # Params
        sample_rate=16000
        n_fft=512
        win_length_ms=25
        hop_length_ms=10
        n_mels=80
        mF=2
        F=27
        mT=5
        pS=0.05
        kernel_size=15
        drop_rate=0.1
        attn_drop_rate=0.0
        max_pos_encoding=10000
        causal=False
        subsampling_filters=180
        dim_model=[180, 256, 360]
        num_heads=4

        # Audio Preprocessing (B, T) -> (B, n_mels, T // hop_length + 1)
        self.audio_preprocessing = preprocessing.AudioPreprocessing(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length_ms=win_length_ms,
            hop_length_ms=hop_length_ms,
            n_mels=n_mels,
            normalize=False,
            mean=-5.6501,
            std=4.2280
        )

        # Spec Augment
        self.spec_augment = preprocessing.SpecAugment(
            mF=mF,
            F=F,
            mT=mT,
            pS=pS
        )

        # Unsqueeze (B, N, T) -> (B, 1, N, T)
        self.unsqueeze = layers.Unsqueeze(dim=1)

        # Stem (B, 1, N, T) -> (B, C, N', T')
        self.subsampling_module = modules.ConvNeuralNetwork(
            dim_input=1,
            dim_layers=subsampling_filters,
            kernel_size=3,
            strides=2,
            norm="BatchNorm2d",
            act_fun="Swish",
            drop_rate=0.0,
            dim=2
        )

        # Reshape (B, C, N, T) -> (B, D, T)
        self.reshape = layers.Reshape(shape=(subsampling_filters * n_mels // 2, -1), include_batch=False)

        # Transpose (B, D, T) -> (B, T, D)
        self.transpose = layers.Transpose(1, 2)

        # Linear Proj
        self.linear = layers.Linear(subsampling_filters * n_mels // 2, dim_model[0])

        # Conformer
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params=[
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}}
            ] if att_type == "regular" else [
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 3, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}},
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 1, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}},
                {"class": "GroupedRelPosMultiHeadSelfAttention", "params": {"num_heads": num_heads, "group_size": 1, "attn_drop_rate": attn_drop_rate, "max_pos_encoding": max_pos_encoding, "causal": causal}}
            ] if att_type == "grouped" else [
                {"class": "RelPosPatch1dMultiHeadAttention", "params": {"num_heads": num_heads, "patch_size": 3, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
                {"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}}
            ],
            conv_params={ "class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    def forward(self, x, lengths):

        # Audio Preprocessing 
        x, lengths = self.audio_preprocessing(x, lengths)

        # Spec Augment
        x = self.spec_augment(x, lengths)

        # Unsqueeze 
        x = self.unsqueeze(x)

        # Stem
        x, lengths = self.subsampling_module(x, lengths)

        # Reshape
        x = self.reshape(x)

        # Transpose
        x = self.transpose(x)

        # Linear Proj
        x = self.linear(x)

        # Conformer
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs

class VisualEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, interctc_blocks=[3, 6, 9], num_blocks=[6, 6], loss_prefix="ctc"):
        super(VisualEfficientConformerEncoder, self).__init__()

        # Params
        dim_model=[256, 360]
        num_heads=4
        kernel_size=15
        drop_rate=0.1
        attn_drop_rate=0.0
        max_pos_encoding=10000

        # Stem 88 -> 44
        # MaxPool 44 -> 22 
        # ResNet 22 -> 11 -> 6 -> 3
        # GlobalPool 3 -> 1
        self.front_end = nn.Sequential(
            modules.ConvNeuralNetwork(
                dim_input=1,
                dim_layers=64,
                kernel_size=(5, 7, 7),
                strides=(1, 2, 2),
                norm="BatchNorm3d",
                act_fun="ReLU",
                drop_rate=0.0,
                dim=3
            ),
            layers.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="same"),
            transforms.VideoToImages(), # (B, C, T, H, W) -> (BT, C, H, W)
            ResNet(include_stem=False, dim_output=dim_model[0], model="ResNet18")
        )

        self.expand_time = transforms.ImagesToVideos()

        # Conformer
        self.back_end = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix=loss_prefix
        )

        # Head
        self.head = layers.Linear(dim_model[-1], vocab_size) if include_head else nn.Identity()

    def forward(self, x, lengths):

        # Frontend
        time = x.shape[2]
        x = self.front_end(x) # (B, C, T, H, W) -> (BT, C)
        x = x.unsqueeze(dim=-1).unsqueeze(dim=-1) # (BT, C) -> (BT, C, 1, 1)
        x = self.expand_time(x, time) # (BT, C, 1, 1) -> (B, C, T, 1, 1)
        x = x.squeeze(dim=-1).squeeze(dim=-1).transpose(1, 2) # (B, C, T, 1, 1) -> (B, T, C)

        # Backend
        x, lengths, interctc_outputs = self.back_end(x, lengths)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs
    
class AudioVisualEfficientConformerEncoder(nn.Module):

    def __init__(self, include_head=True, vocab_size=256, v_interctc_blocks=[3, 6], a_interctc_blocks=[8, 11], f_interctc_blocks=[2]):
        super(AudioVisualEfficientConformerEncoder, self).__init__()

        # Params
        dim_model = 360
        num_blocks = 5
        num_heads = 4
        drop_rate = 0.1
        attn_drop_rate = 0.0
        max_pos_encoding = 10000
        kernel_size = 15
        v_num_blocks = [6, 1]
        a_num_blocks = [5, 6, 1]

        # Visual Encoder
        self.video_encoder = VisualEfficientConformerEncoder(include_head=False, vocab_size=vocab_size, interctc_blocks=v_interctc_blocks, num_blocks=v_num_blocks, loss_prefix="v_ctc")

        # Audio Encoder
        self.audio_encoder = AudioEfficientConformerEncoder(include_head=False, vocab_size=vocab_size, interctc_blocks=a_interctc_blocks, num_blocks=a_num_blocks, loss_prefix="a_ctc")

        # Fusion Module
        self.fusion_module = modules.FusionModule(a_dim_model=dim_model, v_dim_model=dim_model, f_dim_model=dim_model)

        # Audio-visual Encoder
        self.audio_visual_encoder = ConformerInterCTC(
            dim_model=dim_model,
            num_blocks=num_blocks,
            interctc_blocks=f_interctc_blocks,
            vocab_size=vocab_size,
            att_params={"class": "RelPos1dMultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate, "num_pos_embeddings": max_pos_encoding, "weight_init": "default", "bias_init": "default"}},
            conv_params={"class": "Conv1d", "params": {"padding": "same", "kernel_size": kernel_size}},
            ff_ratio=4,
            drop_rate=drop_rate,
            pos_embedding=None, 
            mask=attentions.Mask(), 
            conv_stride=2, 
            batch_norm=True,
            loss_prefix="f_ctc"
        )

        # Head
        self.head = layers.Linear(dim_model, vocab_size) if include_head else nn.Identity()

    def forward(self, video, video_len, audio, audio_len):

        # Visual Encoder
        video, video_len, video_interctc_outputs = self.video_encoder(video, video_len)

        # Audio Encoder
        audio, audio_len, audio_interctc_outputs = self.audio_encoder(audio, audio_len)

        # Fusion Module
        x = self.fusion_module(audio, video)
        lengths = audio_len

        # Audio-visual Encoder
        x, lengths, interctc_outputs = self.audio_visual_encoder(x, lengths)
        interctc_outputs.update(video_interctc_outputs)
        interctc_outputs.update(audio_interctc_outputs)

        # Head
        x = self.head(x)

        return x, lengths, interctc_outputs