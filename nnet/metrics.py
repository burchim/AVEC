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

# Other
import jiwer

###############################################################################
# Metrics
###############################################################################

class Mean(nn.Module):

    def __init__(self, name="mean"):
        super(Mean, self).__init__()
        self.name = name

    def forward(self, y_true, y_pred):

        # Compute mean
        mean = y_pred.mean()

        return mean

class CategoricalAccuracy(nn.Module):

    def __init__(self, ignore_index=-1, dim_argmax=-1, name="acc"):
        super(CategoricalAccuracy, self).__init__()
        self.name = name
        self.dim_argmax = dim_argmax
        self.ignore_index = ignore_index

    def forward(self, y_true, y_pred):

        # ArgMax
        if self.dim_argmax != None:
            y_pred = y_pred.argmax(dim=self.dim_argmax)

        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Accuracy
        acc = torch.where(y_true==y_pred, 1.0, 0.0)

        # Mask Accuracy
        acc = acc * mask

        # Categorical Accuracy
        acc = 100 * acc.sum() / n

        return acc

class CategoricalAccuracyTopK(nn.Module):

    def __init__(self, ignore_index=-1, dim_topk=-1, topk=5, name=None):
        super(CategoricalAccuracyTopK, self).__init__()
        self.name = name if name != None else "topk{}".format(topk)
        self.ignore_index = ignore_index
        self.topk = topk
        self.dim_topk = dim_topk

    def forward(self, y_true, y_pred):

        # Compute Mask
        mask = torch.where(y_true==self.ignore_index, 0.0, 1.0)

        # Reduction
        n = torch.count_nonzero(mask)

        # Element Wise Topk Accuracy
        values, indices = y_pred.topk(self.topk, dim=self.dim_topk, largest=True, sorted=True)
        y_true = y_true.unsqueeze(dim=-1).repeat(1, 1, self.topk)
        acc = torch.where(y_true==indices, 1.0, 0.0).sum(dim=-1)

        # Mask Accuracy
        acc = acc * mask

        # Categorical Accuracy
        acc = 100 * acc.sum() / n

        return acc

class WordErrorRate(nn.Module):

    def __init__(self, name="wer"):
        super(WordErrorRate, self).__init__()
        self.name = name

    def forward(self, targets, outputs):

        # Word Error Rate
        return torch.tensor(100 * jiwer.wer(targets, outputs, standardize=True))

###############################################################################
# Metric Dictionary
###############################################################################

metric_dict = {
    "CategoricalAccuracy": CategoricalAccuracy,
    "WordErrorRate": WordErrorRate
}