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
from torch import nn

class PCA(nn.Module):

    """ Principal Component Analysis (PCA) """

    def __init__(self, n_components):
        super(PCA, self).__init__()

        # Params
        self.n_components = n_components

    def forward(self, x):

        return self.pca(x, n_components=self.n_components)

    def pca(self, X, n_components):
        
        """ PCA linear proj from (N, din) to (N, dout) """

        # Center Data
        X_mean = X - torch.mean(X, axis=0)

        # Compute Covariance Matrix
        cov_mat = torch.matmul(X_mean.transpose(0, 1), X_mean)

        # Compute Eigenvalues and Eigenvectors of the Covariance Matrix
        eigen_values, eigen_vectors = torch.linalg.eigh(cov_mat)

        # Sort Eigenvectors in descending order
        sorted_index = torch.argsort(eigen_values, descending=True)
        sorted_eigenvectors = eigen_vectors[:, sorted_index] # (D, D)

        # Select n components
        eigenvector_subset = sorted_eigenvectors[:, :n_components] # (D, n)

        # Transform Data (B, D) -> (B, n)
        X_reduced = torch.matmul(X_mean, eigenvector_subset)

        return X_reduced