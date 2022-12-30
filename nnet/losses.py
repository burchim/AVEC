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
import torchaudio

###############################################################################
# Losses
###############################################################################

class Reduction(nn.Module):

    def __init__(self, reduction="mean"):
        super(Reduction, self).__init__()

        assert reduction in ["sum", "mean", "mean_batch"]
        self.reduction = reduction

    def forward(self, x, n_elt=None):

        # Reduction
        if self.reduction == "sum":
            x = x.sum()
        elif self.reduction == "mean" and n_elt == None:
            x = x.mean()
        elif self.reduction == "mean" and n_elt != None:
            x = x.sum() / n_elt
        elif self.reduction == "mean_batch":
            x = x.mean(dim=0).sum()

        return x

class MeanLoss(nn.Module):

    def __init__(self, targets_as_sign=True, targets=None, reduction="mean"):
        super(MeanLoss, self).__init__()
        
        self.targets_as_sign = targets_as_sign
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets
        
        # Loss Sign
        if self.targets_as_sign:
            y_pred = torch.where(y == 1, - y_pred, y_pred)

        # Reduction
        loss = self.reduction(y_pred)

        return loss

class HingeLoss(nn.Module):

    """ Hinge Loss: max(magin - x) if y >= 0, max(magin + x) if y < 0 """

    def __init__(self, margin=1.0, targets=None, reduction='mean'):
        super(HingeLoss, self).__init__()

        # Params
        self.margin = margin

        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Compute Loss
        loss = torch.where(y > 0, (self.margin - y_pred).relu(), (self.margin + y_pred).relu())

        # Reduction
        loss = self.reduction(loss)

        return loss

class MeanAbsoluteError(nn.Module):

    """ MAE Loss: abs(y - x) """

    def __init__(self, convert_one_hot=False, one_hot_axis=-1, masked=False, reduction='mean'):
        super(MeanAbsoluteError, self).__init__()

        # Params
        self.convert_one_hot = convert_one_hot
        self.one_hot_axis = one_hot_axis
        self.masked = masked

        # Loss
        self.loss = nn.L1Loss(reduction='none')

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        if self.masked:
            y_pred, mask = outputs
        else:
            y_pred = outputs

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.one_hot_axis)).type(y_pred.dtype)

        # Compute Loss
        loss = self.loss(input=y_pred, target=y)

        # Mask Loss
        if self.masked:
            loss = loss * mask
            N = mask.count_nonzero()
        else:
            N = loss.numel()

        # Reduction
        loss = self.reduction(loss, n_elt=N)

        return loss

class MeanSquaredError(nn.Module):

    """ MSE Loss: (y - x) ** 2 """

    def __init__(self, convert_one_hot=False, axis=-1, targets=None, factor=1.0, reduction='mean'):
        super(MeanSquaredError, self).__init__()

        # Params
        self.convert_one_hot = convert_one_hot
        self.axis = axis
        self.factor = factor

        # Default Targets
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Loss
        self.loss = nn.MSELoss(reduction='none')

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.axis)).type(y_pred.dtype)

        # Compute Loss
        loss = self.factor * self.loss(input=y_pred, target=y)

        # Reduction
        loss = self.reduction(loss)

        return loss

class HuberLoss(nn.Module):

    def __init__(self, convert_one_hot=False, axis=-1, targets=None, delta=1.0, factor=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()

        # Params
        self.convert_one_hot = convert_one_hot
        self.axis = axis
        self.factor = factor

        # Default Targets
        if targets != None:
            self.register_buffer("targets", torch.tensor(targets))
        else:
            self.targets = None

        # Loss
        self.loss = nn.HuberLoss(reduction='none', delta=delta)

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Outputs
        y_pred = outputs

        # Unpack Targets
        if self.targets != None:
            y = self.targets.expand_as(y_pred).to(y_pred.device)
        else:
            y = targets

        # Convert one hot
        if self.convert_one_hot:
            y = F.one_hot(y, num_classes=y_pred.size(self.axis)).type(y_pred.dtype)

        # Compute Loss
        loss = self.factor * self.loss(input=y_pred, target=y)

        # Reduction
        loss = self.reduction(loss)

        return loss

class SoftmaxCrossEntropy(nn.Module):

    def __init__(self, ignore_index=-1, transpose_logits=False, reduction='mean'):
        super(SoftmaxCrossEntropy, self).__init__()

        # Params
        self.transpose_logits = transpose_logits

        # Loss
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

        # Reduction
        self.reduction = Reduction(reduction)

    def forward(self, targets, outputs):

        # Unpack Targets
        y = targets

        # Unpack Outputs
        logits = outputs

        # transpose Logits
        if self.transpose_logits:
            logits = logits.transpose(1, 2)

        # Compute Loss
        loss = self.loss(input=logits, target=y)

        # Reduction
        loss = self.reduction(loss)

        return loss

class CTCLoss(nn.Module):

    def __init__(self, blank=0, reduction="mean", zero_infinity=False, assert_shorter=True):
        super(CTCLoss, self).__init__()

        # mean: Sum Frames + Mean Batch
        # sum: Sum Frames + Sum Batch
        # default: Mean Frames + Mean Batch
        assert reduction in ["mean", "sum", "default"]

        # Loss
        self.loss = nn.CTCLoss(blank=blank, reduction="mean" if reduction == "default" else "none", zero_infinity=zero_infinity)

        # Reduction
        self.reduction = nn.Identity() if reduction == "default" else Reduction(reduction)

        # Params
        self.assert_shorter = assert_shorter

    def forward(self, targets, outputs):

        # Unpack Targets
        y, y_len = targets

        # Unpack Outputs
        logits, logits_len = outputs

        # Assert
        if self.assert_shorter:
            assert (y_len <= logits_len).all(), "logits length shorter than label length: \nlogits_len \n{} \ny_len \n{}".format(logits_len, y_len)

        # Compute Loss
        loss = self.loss(
             log_probs=torch.nn.functional.log_softmax(logits, dim=-1).transpose(0, 1), # (T, B, V)
             targets=y,
             input_lengths=logits_len,
             target_lengths=y_len
        )

        # Reduction
        loss = self.reduction(loss)

        return loss

class RNNTLoss(torchaudio.transforms.RNNTLoss):

    def __init__(self, blank=0, clamp=-1, reduction="mean"):
        super(RNNTLoss, self).__init__(blank=blank, clamp=clamp, reduction=reduction)

    def forward(self, targets, outputs):

        # Unpack Targets (B, U) and (B,)
        y, y_len = targets

        # Unpack Outputs (B, T, U + 1, V) and (B,)
        logits, logits_len = outputs

        # Compute Loss
        loss = super(RNNTLoss, self).forward(
            logits=logits,
            targets=y.int(),
            logit_lengths=logits_len.int(),
            target_lengths=y_len.int()
        )

        return loss

###############################################################################
# Loss Dictionary
###############################################################################

loss_dict = {
    "SoftmaxCrossEntropy": SoftmaxCrossEntropy,
    "CTC": CTCLoss
}