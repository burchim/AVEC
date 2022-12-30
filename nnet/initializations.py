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

import math

# PyTorch
import torch.nn.init as init

###############################################################################
# Initializations
###############################################################################

# N(mean, std)
def normal(tensor, mean=0.0, std=1.0):
    return init.normal_(tensor, mean=mean, std=std)

# U(a, b)
def uniform(tensor, a=0.0, b=1.0):
    return init.uniform_(tensor, a=a, b=b)

# U(-b, b) where b = sqrt(1/fan_in)
def scaled_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, a=math.sqrt(5), mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def scaled_normal_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(3/fan_in)
def lecun_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, nonlinearity="linear", mode=mode)

# N(0, std**2) where std = sqrt(1/fan_in)
def lecun_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, nonlinearity="linear", mode=mode)

# U(-b, b) where b = sqrt(6/fan_in)
def he_uniform_(tensor, mode="fan_in"):
    return init.kaiming_uniform_(tensor, mode=mode)

# N(0, std**2) where std = sqrt(2/fan_in)
def he_normal_(tensor, mode="fan_in"):
    return init.kaiming_normal_(tensor, mode=mode)

# U(-b, b) where b = sqrt(6/(fan_in + fan_out))
def xavier_uniform_(tensor):
    return init.xavier_uniform_(tensor)

# N(0, std**2) where std = sqrt(2/(fan_in + fan_out))
def xavier_normal_(tensor):
    return init.xavier_normal_(tensor)

# N(0.0, 0.02)
def normal_02_(tensor):
    return init.normal_(tensor, mean=0.0, std=0.02)

###############################################################################
# Initialization Dictionary
###############################################################################

init_dict = {
    "uniform": init.uniform_,
    "normal": init.normal_,

    "ones": init.ones_,
    "zeros": init.zeros_,

    "scaled_uniform": scaled_uniform_,
    "scaled_normal": scaled_normal_,

    "lecun_uniform": lecun_uniform_,
    "lecun_normal": lecun_normal_,

    "he_uniform": he_uniform_,
    "he_normal": he_normal_,

    "xavier_uniform": xavier_uniform_,
    "xavier_normal": xavier_normal_,

    "normal_02": normal_02_
}