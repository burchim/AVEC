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
import math

###############################################################################
# Schedulers
###############################################################################

class Scheduler(nn.Module):

    def __init__(self):
        super(Scheduler, self).__init__()

        # Model Step
        self.model_step = torch.tensor(0)

    def step(self):
        self.model_step += 1
        return self.get_val()

    def get_val(self):
        return self.get_val_step(self.model_step)

    def get_val_step(self, step):
        return None

class ConstantScheduler(Scheduler):

    def __init__(self, val):
        super(ConstantScheduler, self).__init__()

        # Scheduler Params
        self.val = val

    def get_val_step(self, step):
        return self.val

class ConstantDecayScheduler(Scheduler):

    def __init__(self, values, decay_steps):
        super(ConstantDecayScheduler, self).__init__()

        # Scheduler Params
        self.values = values # size 1 + n
        self.decay_steps = decay_steps # size n

    def get_val_step(self, step):

        # Compute Value
        val = self.values[0]
        for i, start_step in enumerate(self.decay_steps):
            if step > start_step:
                val = self.values[i + 1]
            else:
                break

        return val

class WarmupConstantDecayScheduler(Scheduler):

    def __init__(self, warmup_steps, values, decay_steps):
        super(WarmupConstantDecayScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.values = values
        self.decay_steps = decay_steps

    def get_val_step(self, step):

        # Warmup Phase
        if step <= self.warmup_steps:
            return step / self.warmup_steps * self.values[0]

        # Compute Value
        val = self.values[0]
        for i, start_step in enumerate(self.decay_steps):
            if step > start_step:
                val = self.values[i + 1]
            else:
                break

        return val

class LinearDecayScheduler(Scheduler):

    def __init__(self, value_start, value_end, decay_steps):
        super(LinearDecayScheduler, self).__init__()

        # Scheduler Params
        self.value_start = value_start
        self.value_end = value_end
        self.decay_steps = decay_steps

    def get_val_step(self, step):

        # Compute Value
        if step >= self.decay_steps:
            val = self.value_end
        else:
            val = self.value_start - step * (self.value_start - self.value_end) / self.decay_steps

        return val

class NoamDecayScheduler(Scheduler):

    def __init__(self, warmup_steps, dim_decay, val_factor):
        super(NoamDecayScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.dim_decay = dim_decay
        self.val_factor = val_factor

    def get_val_step(self, step):

        # Compute Value
        arg1 = step * (self.warmup_steps**-1.5) # Warmup phase
        arg2 = step**-0.5 # Decay phase
        val = self.val_factor * self.dim_decay**-0.5 * min(arg1, arg2)

        return val

class ExpDecayScheduler(Scheduler):

    def __init__(self, warmup_steps, val_max, alpha, end_step):
        super(ExpDecayScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.val_max = val_max
        self.alpha = alpha
        self.end_step = end_step

    def get_val_step(self, step):

        # Compute Value
        arg1 = step / self.warmup_steps * self.val_max # Warmup phase
        arg2 = self.val_max * self.alpha**((step - self.warmup_steps) / (self.end_step - self.warmup_steps)) # Decay phase
        val = min(arg1, arg2)

        return val

class CosineAnnealingScheduler(Scheduler):

    def __init__(self, warmup_steps, val_max, val_min, end_step):
        super(CosineAnnealingScheduler, self).__init__()

        # Scheduler Params
        self.warmup_steps = warmup_steps
        self.val_max = val_max
        self.val_min = val_min
        self.end_step = end_step

    def get_val_step(self, step):

        # Compute LR
        if step <= self.warmup_steps: # Warmup phase
            val = step / self.warmup_steps * self.val_max
        elif step <= self.end_step: # Annealing phase
            val = (self.val_max - self.val_min) * 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / (self.end_step - self.warmup_steps))) + self.val_min
        else: # End phase
            val = self.val_min

        return val

###############################################################################
# Scheduler Dictionary
###############################################################################

scheduler_dict = {
    "Constant": ConstantScheduler,
    "ConstantDecay": ConstantDecayScheduler,
    "NoamDecay": NoamDecayScheduler,
    "ExpDecay": ExpDecayScheduler,
    "CosineAnnealing": CosineAnnealingScheduler
}