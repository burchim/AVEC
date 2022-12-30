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

# Neural Nets
from nnet import layers
from nnet import embeddings

###############################################################################
# Multi-Head Attention Layers
###############################################################################

class MultiHeadAttention(nn.Module):

    """Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads

    References: 
        Attention Is All You Need, Vaswani et al.
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, dim_model, num_heads, attn_drop_rate, weight_init="scaled_uniform", bias_init="zeros", output_proj=True, dim_kv=None):
        super(MultiHeadAttention, self).__init__()

        # Dim Key Value
        if dim_kv == None:
            dim_kv = dim_model

        # Attention Params
        self.num_heads = num_heads # H
        self.dim_model = dim_model # D
        self.dim_head = dim_model // num_heads # d
        self.output_proj = output_proj
        self.dim_kv = dim_kv

        # Attention Dropout
        self.dropout = layers.Dropout(attn_drop_rate) if attn_drop_rate > 0 else nn.Identity()

        # Init Layers
        self.init_layers(weight_init, bias_init)

    def init_layers(self, weight_init, bias_init):

        # Linear Layers
        self.query_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.key_layer = layers.Linear(self.dim_kv, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.value_layer = layers.Linear(self.dim_kv, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.output_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init) if self.output_proj else nn.Identity()

    def forward_inputs(self, Q, K, V):

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        return Q, K, V

    def forward_outputs(self, O):

        # Linear Layers
        O = self.output_layer(O)

        return O

    def forward(self, x, mask=None, return_att_w=False):
        return self.forwardQKV(x, x, x, mask, return_att_w)

    def forwardQKV(self, Q, K, V, mask=None, return_att_w=False):

        """Scaled Dot-Product Multi-Head Attention

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, T)

        """

        # Batch size B
        batch_size = Q.size(0)

        # Input Linear Layers
        Q, K, V = self.forward_inputs(Q, K, V)

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Att scores (B, H, T, T)
        att_scores = Q.matmul(K.transpose(2, 3)) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        return (O, att_w.detach()) if return_att_w else O

    def pad(self, Q, K, V, mask, chunk_size):

        # Compute Overflows
        overflow_Q = Q.size(1) % chunk_size
        overflow_KV = K.size(1) % chunk_size
        
        padding_Q = chunk_size - overflow_Q if overflow_Q else 0
        padding_KV = chunk_size - overflow_KV if overflow_KV else 0

        batch_size, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0)

        # Update Padding Mask
        if mask is not None:

            # (B, 1, 1, T) -> (B, 1, 1, T + P) 
            if mask.size(2) == 1:
                mask = F.pad(mask, pad=(0, padding_KV), value=0)
            # (B, 1, T, T) -> (B, 1, T + P, T + P)
            else:
                mask = F.pad(mask, pad=(0, padding_Q, 0, padding_KV), value=0)

        elif padding_KV:

            # None -> (B, 1, 1, T + P) 
            mask = F.pad(Q.new_zeros(batch_size, 1, 1, seq_len_KV), pad=(0, padding_KV), value=0)

        return Q, K, V, mask, padding_Q

class NdMultiHeadAttention(MultiHeadAttention):

    """ Flatten Nd before Attention """

    def __init__(self, dim_model, num_heads, attn_drop_rate, weight_init="scaled_uniform", bias_init="zeros", output_proj=True):
        super(NdMultiHeadAttention, self).__init__(dim_model=dim_model, num_heads=num_heads, attn_drop_rate=attn_drop_rate, weight_init=weight_init, bias_init=bias_init, output_proj=output_proj)

    def init_layers(self, weight_init, bias_init):

        # Linear Layers
        self.query_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.key_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.value_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init)
        self.output_layer = layers.Linear(self.dim_model, self.dim_model, weight_init=weight_init, bias_init=bias_init) if self.output_proj else nn.Identity()

    def forward_inputs(self, Q, K, V):

        # Record Shape
        self.shape = Q.shape

        # Flatten (B, ..., C) -> (B, N, C)
        Q = Q.flatten(start_dim=1, end_dim=-2)
        K = K.flatten(start_dim=1, end_dim=-2)
        V = V.flatten(start_dim=1, end_dim=-2)

        # Linear Layers
        Q = self.query_layer(Q)
        K = self.key_layer(K)
        V = self.value_layer(V)

        return Q, K, V

    def forward_outputs(self, O):

        # Linear Layers
        O = self.output_layer(O)

        # Reshape (B, N, C) -> (B, ..., C)
        O = O.reshape(self.shape)

        return O

class RelPos1dMultiHeadAttention(MultiHeadAttention):

    """ Relative Position 1d Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        max_pos_embeddings: maximum position encodings E

    """
    
    def __init__(self, dim_model, num_heads, num_pos_embeddings, attn_drop_rate, weight_init="scaled_uniform", bias_init="zeros", output_proj=True, causal=False):
        super(RelPos1dMultiHeadAttention, self).__init__(dim_model=dim_model, num_heads=num_heads, attn_drop_rate=attn_drop_rate, weight_init=weight_init, bias_init=bias_init, output_proj=output_proj)
     
        # Relative Positional Embeddings
        self.causal = causal
        self.rel_pos_enc = embeddings.RelativeSinusoidalPositionalEncoding(num_pos_embeddings, self.dim_model, self.causal)
        self.pos_layer = layers.Linear(self.dim_model, self.dim_model)

    def rel_to_abs(self, att_scores):

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores

    def forwardQKV(self, Q, K, V, mask=None, return_att_w=False):

        # Batch Size
        batch_size = Q.size(0)

        # Input Linear Layers (B, T, D) -> (B, T, D)
        Q, K, V = self.forward_inputs(Q, K, V)

        # Relative Positional Embeddings (B, 2*T-1, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, h, T, d)
        Q = Q.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Reshape and Transpose (B, 2*T'-1, D) -> (B, h, 2*T'-1, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, h, T, T)
        att_scores_K = Q.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Q.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, h, T, T)
        att_w = att_scores.softmax(dim=-1)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, h, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, h, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        return (O, att_w.detach()) if return_att_w else O

class RelPosPatch1dMultiHeadAttention(RelPos1dMultiHeadAttention):

    """ Relative Position Patch 1d Mutli-Head Attention Layer

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        pool_size: attention pooling size (Pt, Ph, Pw)
        max_pos_embeddings: downsampled maximum position encodings E

    """

    def __init__(self, dim_model, num_heads, patch_size, num_pos_embeddings, attn_drop_rate, weight_init="scaled_uniform", bias_init="zeros", output_proj=True):
        super(RelPosPatch1dMultiHeadAttention, self).__init__(dim_model=dim_model, num_heads=num_heads, attn_drop_rate=attn_drop_rate, num_pos_embeddings=num_pos_embeddings, weight_init=weight_init, bias_init=bias_init, output_proj=output_proj)

        # Attention Params
        self.patch_size = patch_size
        self.downsample = layers.AvgPool1d(kernel_size=patch_size, stride=patch_size, channels_last=True)
        self.upsample = layers.Upsample(scale_factor=patch_size, mode="nearest", channels_last=True)

        # Mask Pool
        self.mask_pool = nn.MaxPool1d(kernel_size=patch_size, stride=patch_size)

    def forwardQKV(self, Q, K, V, mask=None, return_att_w=False):

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.patch_size)

        # Mask Pooling
        if mask != None:

            # Min Pool mask
            mask = mask.squeeze(dim=1) # (1 or B, 1, N, N) -> (1 or B, N, N)
            mask = - self.mask_pool(- mask) # (1 or B, N, N) -> (1 or B, N, N / P)
            mask = mask.transpose(1, 2)
            mask = - self.mask_pool(- mask) # (1 or B, N / P, N) -> (1 or B, N / P, N / P)
            mask = mask.transpose(1, 2)
            mask = mask.unsqueeze(dim=1) # (1 or B, N / P, N / P) -> (1 or B, 1, N / P, N / P)

        # AvgPool1d (B, T, D) -> (B, T/Pt, D)
        Q = self.downsample(Q)
        K = self.downsample(K)
        V = self.downsample(V)

        # Rel Pos 1d Multi-Head Attention
        O = super(RelPosPatch1dMultiHeadAttention, self).forwardQKV(Q, K, V, mask=mask, return_att_w=return_att_w)

        # Extract att_w
        if return_att_w:
            O, att_w = O

        # UpSample (B, T/Pt, D) -> (B, T, D)
        O = self.upsample(O)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        return (O, att_w) if return_att_w else O

class RelPosMultiHeadSelfAttention(MultiHeadAttention):

    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References: 
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(self, dim_model, num_heads, attn_drop_rate, max_pos_encoding, weight_init="scaled_uniform", bias_init="zeros", output_proj=True, causal=False):
        super(RelPosMultiHeadSelfAttention, self).__init__(dim_model=dim_model, num_heads=num_heads, attn_drop_rate=attn_drop_rate, weight_init=weight_init, bias_init=bias_init, output_proj=output_proj)

        # Position Embedding Layer
        self.pos_layer = layers.Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
        nn.init.zeros_(self.u)
        self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
        nn.init.zeros_(self.v)

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = embeddings.RelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.causal)

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References: 
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
            
            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores

    def forwardQKV(self, Q, K, V, mask=None, return_att_w=False, hidden=None):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Batch size B
        batch_size = Q.size(0)

        # Input Linear Layers
        Q, K, V = self.forward_inputs(Q, K, V)

        # Hidden State Provided
        if hidden:
            K = torch.cat([hidden["K"], K], dim=1)
            V = torch.cat([hidden["V"], V], dim=1)

        # Update Hidden State
        hidden = {"K": K.detach(), "V": V.detach()}

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T, Th + T)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T, Th + T)
        att_w = att_scores.softmax(dim=-1)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, H, T, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        return (O, att_w.detach(), hidden) if return_att_w else O

class GroupedRelPosMultiHeadSelfAttention(RelPosMultiHeadSelfAttention):

    """Grouped Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements
        group_size: attention group size

    """

    def __init__(self, dim_model, num_heads, attn_drop_rate, max_pos_encoding, group_size, causal, weight_init="scaled_uniform", bias_init="zeros", output_proj=True):
        super(GroupedRelPosMultiHeadSelfAttention, self).__init__(dim_model=dim_model, num_heads=num_heads, attn_drop_rate=attn_drop_rate, max_pos_encoding=max_pos_encoding, causal=causal, weight_init=weight_init, bias_init=bias_init, output_proj=output_proj)

        # Attention Params
        self.group_size = group_size # G
        self.dim_head = (self.group_size * dim_model) // self.num_heads # d

        # Grouped Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = embeddings.GroupedRelativeSinusoidalPositionalEncoding(max_pos_encoding, self.dim_model, self.group_size, self.causal)

    def forwardQKV(self, Q, K, V, mask=None, return_att_w=False, hidden=None):

        # Batch size B
        batch_size = Q.size(0)

        # Input Linear Layers
        Q, K, V = self.forward_inputs(Q, K, V)

        # Hidden State Provided
        if hidden:
            Kh = torch.cat([hidden["K"], K], dim=1)
            Vh = torch.cat([hidden["V"], V], dim=1)
            K = torch.cat([hidden["K"][:, hidden["K"].size(1)%self.group_size:], K], dim=1)
            V = torch.cat([hidden["V"][:, hidden["V"].size(1)%self.group_size:], V], dim=1)

            # Update Hidden State
            hidden = {"K": Kh.detach(), "V": Vh.detach()}

        else:

            # Update Hidden State
            hidden = {"K": K.detach(), "V": V.detach()}

        # Chunk Padding
        Q, K, V, mask, padding = self.pad(Q, K, V, mask, chunk_size=self.group_size)

        # Add Bias
        Qu = Q + self.u
        Qv = Q + self.v

        # Relative Positional Embeddings (B, Th + 2*T-G, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T//G, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        # Reshape and Transpose (B, Th + 2*T-G, D) -> (B, H, Th//G + 2*T//G-1, d) / (B, Th + T, D) -> (B, H, Th//G + T//G, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # att_scores (B, H, T//G, Th//G + T//G)
        att_scores_K = Qu.matmul(K.transpose(2, 3))
        att_scores_E = self.rel_to_abs(Qv.matmul(E.transpose(2, 3)))
        att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

        # Apply mask
        if mask is not None:

            # Slice Mask (B, 1, T, T) -> (B, 1, T//G, T//G)
            mask = mask[:, :, ::self.group_size, ::self.group_size]

            # Apply mask
            att_scores += (mask.logical_not() * -1e9)

        # Att weights (B, H, T//G, Th//G + T//G)
        att_w = att_scores.softmax(dim=-1)

        # Att output (B, H, T//G, d)
        O = att_w.matmul(V)

        # Transpose and Reshape (B, H, T//G, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Slice Padding
        O = O[:, :O.size(1) - padding]

        # Output linear layer
        O = self.forward_outputs(O)

        return (O, att_w.detach(), hidden) if return_att_w else O

###############################################################################
# Attention Masks
###############################################################################

class Mask(nn.Module):

    """ Binary Mask (0 = masked, 1 = unmasked)
    
    Args:
        left_context: number of elements to mask on left context
        right_context: number of elements to mask on right context (0 for causal mask)
        seq_len_axis: tensor axes to compute sequence length (1 for sequences, 1 2 for images, 1 2 3 for vieos ...)
        mask_start: mask starting position in tokens
        unsqueeze_head: unsqueeze mask for Multi Head Attention

    Forward:
        x: tensor to compute mask
        x_len: tensor lengths to compute padding mask (if provided)

    """

    def __init__(self, left_context=None, right_context=None, seq_len_axis=1, mask_start=0, unsqueeze_head=True):
        super(Mask, self).__init__()

        self.left_context = left_context
        self.right_context = right_context
        self.seq_len_axis = [seq_len_axis] if isinstance(seq_len_axis, int) else seq_len_axis
        self.mask_start = mask_start
        self.unsqueeze_head = unsqueeze_head

    def padding_mask(self, seq_len, x_len):

        # Init Mask (B, T)
        mask = x_len.new_zeros(x_len.size(0), seq_len)

        # Batch Loop
        for b in range(x_len.size(0)):
            mask[b, :x_len[b]] = x_len.new_ones(x_len[b])

        # Padding Mask (B, 1, T)
        return mask[:, None, :]

    def forward(self, x, x_len=None):

        # Seq Length T
        seq_len = torch.prod(torch.tensor([x.size(axis) for axis in self.seq_len_axis]))

        # Right Context Mask (T, T)
        right_context_mask = x.new_ones(seq_len, seq_len)
        if self.right_context != None:
            right_context_mask = right_context_mask.tril(diagonal=self.right_context)

        # Left Context Mask (T, T)
        left_context_mask = x.new_ones(seq_len, seq_len)
        if self.left_context != None:
            left_context_mask = left_context_mask.triu(diagonal=-self.left_context)

        # Full Context Mask (T, T)
        context_mask = right_context_mask.minimum(left_context_mask)

        # Mask Start
        context_mask[:self.mask_start, :self.mask_start] = 1

        # Padding Mask
        if x_len is not None:

            # Padding Mask (B, 1, T)
            padding_mask = self.padding_mask(seq_len, x_len)

            # Context Mask Union Padding Mask (B, T, T)
            context_mask = context_mask.minimum(padding_mask)
        
        else:

            # Unsqueeze Batch (1, T, T)
            context_mask = context_mask[None, :, :]

        # Unsqueeze Head (B or 1, 1, T, T)
        if self.unsqueeze_head:
            context_mask = context_mask[:, None, :, :]

        return context_mask

###############################################################################
# Attention Dictionary
###############################################################################

att_dict = {
    "MultiHeadAttention": MultiHeadAttention,
    "NdMultiHeadAttention": NdMultiHeadAttention,
    "RelPos1dMultiHeadAttention": RelPos1dMultiHeadAttention,
    "RelPosPatch1dMultiHeadAttention": RelPosPatch1dMultiHeadAttention,
    "RelPosMultiHeadSelfAttention": RelPosMultiHeadSelfAttention,
    "GroupedRelPosMultiHeadSelfAttention": GroupedRelPosMultiHeadSelfAttention
}