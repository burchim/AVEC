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
from nnet.model import Model
from nnet import models
from nnet import networks
from nnet import losses
from nnet import optimizers
from nnet import schedulers
from nnet import embeddings
from nnet import layers
from nnet import optimizers
from nnet import attentions
from nnet import losses
from nnet import metrics

class VisualEfficientConformerCE(Model):

    def __init__(self, vocab_size=500):
        super(VisualEfficientConformerCE, self).__init__(name="Visual Efficient Conformer CE")

        self.encoder = networks.VisualEfficientConformerEncoder(vocab_size=vocab_size, interctc_blocks=[])

    def forward(self, inputs):
        return self.encoder(inputs, lengths=None)[0].mean(dim=1)

    def compile(
        self, 
        losses=losses.SoftmaxCrossEntropy(),
        loss_weights=None,
        optimizer="Adam",
        metrics=metrics.CategoricalAccuracy(),
        decoders=None
    ):

        if optimizer == "Adam":
            lr = schedulers.NoamDecayScheduler(warmup_steps=10000, dim_decay=360, val_factor=2)
            optimizer = optimizers.Adam(params=self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)

        super(VisualEfficientConformerCE, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )
            
class AudioEfficientConformerInterCTC(Model):

    def __init__(self, vocab_size=256, att_type="patch", interctc_blocks=[3, 6, 10, 13]):
        super(AudioEfficientConformerInterCTC, self).__init__(name="Audio Efficient Conformer Inter CTC")

        self.encoder = networks.AudioEfficientConformerEncoder(vocab_size=vocab_size, att_type=att_type, interctc_blocks=interctc_blocks)

    def forward(self, inputs):
        x, lengths = inputs
        x, lengths, interctc_outputs = self.encoder(x, lengths)
        outputs = {"outputs": [x, lengths]}
        outputs.update(interctc_outputs)
        return outputs

    def compile(
        self, 
        losses=losses.CTCLoss(),
        loss_weights=[0.5/4, 0.5/4, 0.5/4, 0.5/4, 0.5],
        optimizer="Adam",
        metrics=None,
        decoders=None
    ):

        if optimizer == "Adam":
            lr = schedulers.NoamDecayScheduler(warmup_steps=10000, dim_decay=360, val_factor=2)
            optimizer = optimizers.Adam(params=self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)

        super(AudioEfficientConformerInterCTC, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )

class VisualEfficientConformerInterCTC(Model):

    def __init__(self, vocab_size=256, interctc_blocks=[3, 6, 9], test_augments=None):
        super(VisualEfficientConformerInterCTC, self).__init__(name="Visual Efficient Conformer Inter CTC")

        self.encoder = networks.VisualEfficientConformerEncoder(vocab_size=vocab_size, interctc_blocks=interctc_blocks)
        self.test_augments = test_augments if isinstance(test_augments, list) else [test_augments] if test_augments is not None else test_augments

    def forward(self, inputs):
        video, video_lengths = inputs
        x, lengths, interctc_outputs = self.encoder(video.permute(0, 4, 1, 2, 3), video_lengths)

        assert not (self.training and self.test_augments is not None), "Training requires setting test_time_aug to False / test_augments to None"

        # Test Augment
        if not self.training and self.test_augments is not None:
            x_list = [x]
            lengths_list = [lengths]
            for test_augment in self.test_augments:
                x_aug, lengths_aug, interctc_outputs_aug = self.encoder(test_augment(video.permute(0, 4, 1, 2, 3)), video_lengths)
                x_list.append(x_aug)
                lengths_list.append(lengths_aug)
            x = torch.stack(x_list, dim=1)
            lengths = torch.stack(lengths_list, dim=1)

        outputs = {"outputs": [x, lengths]}
        outputs.update(interctc_outputs)
        return outputs

    def compile(
        self, 
        losses=losses.CTCLoss(),
        loss_weights=[0.5/3, 0.5/3, 0.5/3, 0.5],
        optimizer="Adam",
        metrics=None,
        decoders=None
    ):

        if optimizer == "Adam":
            lr = schedulers.NoamDecayScheduler(warmup_steps=10000, dim_decay=360, val_factor=2)
            optimizer = optimizers.Adam(params=self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)

        super(VisualEfficientConformerInterCTC, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )

class AudioVisualEfficientConformerInterCTC(Model):

    def __init__(self, vocab_size=256, v_interctc_blocks=[3, 6], a_interctc_blocks=[8, 11], f_interctc_blocks=[2]):
        super(AudioVisualEfficientConformerInterCTC, self).__init__(name="Audio-Visual Efficient Conformer Inter CTC")

        self.encoder = networks.AudioVisualEfficientConformerEncoder(vocab_size=vocab_size, v_interctc_blocks=v_interctc_blocks, a_interctc_blocks=a_interctc_blocks, f_interctc_blocks=f_interctc_blocks)

    def forward(self, inputs):
        video, video_len, audio, audio_len = inputs
        x, lengths, interctc_outputs = self.encoder(video.permute(0, 4, 1, 2, 3), video_len, audio, audio_len)
        outputs = {"outputs": [x, lengths]}
        outputs.update(interctc_outputs)
        return outputs

    def compile(
        self, 
        losses=losses.CTCLoss(),
        loss_weights={"v_ctc_2": 0.5 / 3, "v_ctc_5": 0.5 / 3, "a_ctc_7": 0.5 / 3, "a_ctc_10": 0.5 / 3, "f_ctc_1": 0.5 / 3, "outputs": 0.5},
        optimizer="Adam",
        metrics=None,
        decoders=None
    ):

        if optimizer == "Adam":
            lr = schedulers.NoamDecayScheduler(warmup_steps=10000, dim_decay=360, val_factor=2)
            optimizer = optimizers.Adam(params=self.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)

        super(AudioVisualEfficientConformerInterCTC, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )
        
class GPT(models.Classifier):

    """ GPT-3

    Reference:
    "Language Models are Few-Shot Learners", Brown et al.
    https://arxiv.org/abs/2005.14165
    
    """

    def __init__(self, vocab_size=25000, padding_idx=None, max_pos_encoding=2048, model="GPT-Small", pos_embedding=embeddings.PosEmbedding1d, drop_rate=0.1):
        super(GPT, self).__init__(name=model)

        assert model in ["GPT-Small", "GPT-Medium", "GPT-Large", "GPT-XL", "GPT-2.7B", "GPT-6.7B", "GPT-13.0B", "GPT-175.0B"]

        if model == "GPT-Small":
            dim_model = 768
            num_blocks = 12
            num_heads = 12
        elif model == "GPT-Medium":
            dim_model = 1024
            num_blocks = 24
            num_heads = 16
        elif model == "GPT-Large":
            dim_model = 1536
            num_blocks = 24
            num_heads = 16
        elif model == "GPT-XL":
            dim_model = 2048
            num_blocks = 24
            num_heads = 24
        elif model == "GPT-2.7B":
            dim_model = 2560
            num_blocks = 32
            num_heads = 32
        elif model == "GPT-6.7B":
            dim_model = 4096
            num_blocks = 32
            num_heads = 32
        elif model == "GPT-13.0B":
            dim_model = 5140
            num_blocks = 40
            num_heads = 40
        elif model == "GPT-175.0B":
            dim_model = 12288
            num_blocks = 96
            num_heads = 96

        # Default Params
        ff_ratio = 4
        emb_drop_rate = drop_rate
        drop_rate = drop_rate
        attn_drop_rate = drop_rate
        act_fun="GELU"

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim_model,
            padding_idx=padding_idx
        )

        self.transformer = networks.Transformer(
            dim_model=dim_model,
            num_blocks=num_blocks,
            att_params={"class": "MultiHeadAttention", "params": {"num_heads": num_heads, "attn_drop_rate": attn_drop_rate}},
            ff_ratio=ff_ratio,
            emb_drop_rate=emb_drop_rate,
            drop_rate=drop_rate,
            act_fun=act_fun,
            pos_embedding=pos_embedding(num_embeddings=max_pos_encoding, dim_emb=dim_model),
            inner_dropout=False,
            mask=attentions.Mask(right_context=0)
        )

        self.head = layers.Linear(
            in_features=dim_model,
            out_features=vocab_size
        )

        def init_weights(m, N):

            if isinstance(m, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.ones_(m.weight)

        self.apply(lambda m: init_weights(m, num_blocks))
    
    def compile(self, losses=losses.SoftmaxCrossEntropy(transpose_logits=True), loss_weights=None, optimizer="AdamW", metrics={"output": [metrics.CategoricalAccuracy(), metrics.CategoricalAccuracyTopK(topk=10)]}, decoders=None):

        if optimizer == "AdamW":

            # All models were trained for a total of 300 billion tokens.
            warmup_steps = 750 # 375e6 warmup tokens with 0.5M tokens / step
            end_step = 520000 # 260e9 cosine tokens with 0.5M tokens / step
            # 300e9 total tokens: 600000 steps with 0.5M tokens / step
            
            if self.name == "GPT-Small":
                lr_max = 6e-4
                lr_min = 6e-5
            elif self.name == "GPT-Medium":
                lr_max = 3e-4
                lr_min = 3e-5
            elif self.name == "GPT-Large":
                lr_max = 2.5e-4
                lr_min = 2.5e-5
            elif self.name == "GPT-XL":
                lr_max = 2e-4
                lr_min = 2e-5
            elif self.name == "GPT-2.7B":
                lr_max = 1.6e-4
                lr_min = 1.6e-5
            elif self.name == "GPT-6.7B":
                lr_max = 1.2e-4
                lr_min = 1.2e-5
            elif self.name == "GPT-13.0B":
                lr_max = 1e-4
                lr_min = 1e-5
            elif self.name == "GPT-175.0B":
                lr_max = 0.6e-4
                lr_min = 0.6e-5

            optimizer = optimizers.AdamW(params=optimizers.get_decay_param_groups(self, weight_decay=0.1), lr=schedulers.CosineAnnealingScheduler(warmup_steps=warmup_steps, val_max=lr_max, val_min=lr_min, end_step=end_step), betas=(0.9, 0.95), eps=1e-8)

        super(GPT, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = self.head(x)

        return x