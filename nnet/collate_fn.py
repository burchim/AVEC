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

###############################################################################
# Collate Functions
###############################################################################

class Collate(nn.Module):

    def __init__(self):
        super(Collate, self).__init__()

    def forward(self, samples):
        return samples

class CollateFn(nn.Module):

    """ Collate samples to List / Dict

    Args:
        - inputs_params_: List / Dict of collate param for inputs
        - targets_params: List / Dict of collate param for targets

    Collate Params Dict:
        - axis: axis to select samples
        - padding: whether to pad samples
        - padding_value: padding token, default 0

    """

    def __init__(self, inputs_params=[{"axis": 0}], targets_params=[{"axis": 1}]):
        super(CollateFn, self).__init__()

        assert isinstance(inputs_params, dict) or isinstance(inputs_params, list) or isinstance(inputs_params, tuple)
        self.inputs_params = inputs_params
        assert isinstance(targets_params, dict) or isinstance(targets_params, list) or isinstance(targets_params, tuple)
        self.targets_params = targets_params

        # Default Params
        if  isinstance(inputs_params, dict):
            for params in self.inputs_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

            for params in self.targets_params.values():
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

        else:
            for params in self.inputs_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

            for params in self.targets_params:
                if not "padding" in params:
                    params["padding"] = False
                if not "padding_value" in params:
                    params["padding_value"] = 0
                if not "start_token" in params:
                    params["start_token"] = None
                if not "end_token" in params:
                    params["end_token"] = None

    def forward(self, samples):
        return {"inputs": self.collate(samples, self.inputs_params), "targets": self.collate(samples, self.targets_params)}

    def collate(self, samples, collate_params):

        # Dict
        if isinstance(collate_params, dict):
            collates = {}
            for name, params in collate_params.items():

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates[name] = collate
        # List
        elif isinstance(collate_params, list):
            collates = []
            for params in collate_params:

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates.append(collate)
        # Tuple
        elif isinstance(collate_params, tuple):
            collates = []
            for params in collate_params:

                # Select
                collate = [sample[params["axis"]] for sample in samples]

                # Start Token
                if params["start_token"]:
                    collate = [torch.cat([params["start_token"] * item.new_ones(1), item]) for item in collate]

                # End Token
                if params["end_token"]:
                    collate = [torch.cat([item, params["end_token"] * item.new_ones(1)]) for item in collate]

                # Padding
                if params["padding"]:
                    collate = torch.nn.utils.rnn.pad_sequence(collate, batch_first=True, padding_value=params["padding_value"])
                else:
                    collate = torch.stack(collate, axis=0)

                # Append
                collates.append(collate)
            collates = tuple(collates)

        collates = collates[0] if len(collates) == 1 else collates

        return collates