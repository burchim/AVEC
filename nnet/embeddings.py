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
from torch.nn.modules.utils import _single

class SinPosEmbedding(nn.Module):

    def __init__(self, num_embeddings, dim_emb):
        super(SinPosEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.dim_emb = dim_emb

        pos_encoding = torch.zeros(self.num_embeddings, self.dim_emb)
        pos = torch.arange(0, self.num_embeddings, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, self.dim_emb // 2, dtype=torch.float).unsqueeze(0) 
        angles = pos / 10000**(2 * i / self.dim_emb)

        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, x):

        # Apply Embeddings (B, T, D)
        x = x + self.pos_encoding[:, :x.shape[1]]

        return x

class PosEmbedding1d(nn.Module):

    def __init__(self, num_embeddings, dim_emb):
        super(PosEmbedding1d, self).__init__()

        self.num_embeddings = _single(num_embeddings)
        self.dim_emb = dim_emb
        self.pos_encoding = nn.Parameter(torch.zeros(self.num_embeddings + (self.dim_emb,)))

    def forward(self, x):

        assert x.dim() == 3, "input must be (Batch, Length, Features)"
        
        # Apply Embeddings
        x = x + self.pos_encoding[:x.shape[-2]]

        return x

class SinusoidalPositionalEncoding(nn.Module):
    
    """

    Sinusoidal Positional Encoding

    Reference: "Attention Is All You Need" by Vaswani et al.
    https://arxiv.org/abs/1706.03762

    """

    def __init__(self, max_len, dim_model):
        super(SinusoidalPositionalEncoding, self).__init__()

        pos_encoding = torch.zeros(max_len, dim_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        i = torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) 
        angles = pos / 10000**(2 * i / dim_model)

        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)

    def forward(self, batch_size=1, seq_len=None):

        # (B, T, D)
        if seq_len is not None:
            P = self.pos_encoding[:, :seq_len]

        # (B, Tmax, D)
        else:
            P = self.pos_encoding

        return P.repeat(batch_size, 1, 1)

class RelativeSinusoidalPositionalEncoding(nn.Module):
    
    """
        Relative Sinusoidal Positional Encoding

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - 1
    """

    def __init__(self, max_len, dim_model, causal=False):
        super(RelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - 1, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len-1, end=0, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len]

            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:,:self.max_len]

        # Full Context
        else:

            # (B, Th + 2*T-1, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len - 1  + seq_len]
            
            # (B, 2*Tmax-1, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)

class GroupedRelativeSinusoidalPositionalEncoding(nn.Module):
    
    """
        Relative Sinusoidal Positional Encoding for grouped multi-head attention

        Positional encoding for left context (sin) and right context (cos)
        Total context = 2 * max_len - group_size
    """

    def __init__(self, max_len, dim_model, group_size=1, causal=False):
        super(GroupedRelativeSinusoidalPositionalEncoding, self).__init__()

        # PE
        pos_encoding = torch.zeros(2 * max_len - group_size % 2, dim_model)

        # Positions (max_len - 1, ..., max_len - 1)
        pos_left = torch.arange(start=max_len-1, end=group_size % 2 - 1, step=-1, dtype=torch.float)
        pos_right = torch.arange(start=0, end=-max_len, step=-1, dtype=torch.float)
        pos = torch.cat([pos_left, pos_right], dim=0).unsqueeze(1)

        # Angles
        angles = pos / 10000**(2 * torch.arange(0, dim_model // 2, dtype=torch.float).unsqueeze(0) / dim_model)

        # Rel Sinusoidal PE
        pos_encoding[:, 0::2] = angles.sin()
        pos_encoding[:, 1::2] = angles.cos()

        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding, persistent=False)
        self.max_len = max_len
        self.causal = causal
        self.group_size = group_size

    def forward(self, batch_size=1, seq_len=None, hidden_len=0):

        # Causal Context
        if self.causal:

            # (B, Th + T, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len - hidden_len : self.max_len]
            
            # (B, Tmax, D)
            else:
                R = self.pos_encoding[:,:self.max_len]
        else:

            # (B, Th + 2*T-G, D)
            if seq_len is not None:
                R = self.pos_encoding[:, self.max_len - seq_len + self.group_size // 2 - hidden_len : self.max_len - self.group_size % 2  + seq_len - self.group_size // 2 ]
            
            # (B, 2*Tmax-G, D)
            else:
                R = self.pos_encoding

        return R.repeat(batch_size, 1, 1)