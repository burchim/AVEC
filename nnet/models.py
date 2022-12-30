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

# NeuralNets
from nnet.model import Model
from nnet import losses
from nnet import metrics

###############################################################################
# Models
###############################################################################

class Classifier(Model):

    def __init__(self, name="Classifier"):
        super(Classifier, self).__init__(name=name)

    def compile(
        self, 
        losses=losses.SoftmaxCrossEntropy(),
        loss_weights=None,
        optimizer="Adam",
        metrics=metrics.CategoricalAccuracy(),
        decoders=None
    ):

        super(Classifier, self).compile(
            losses=losses,
            loss_weights=loss_weights,
            optimizer=optimizer,
            metrics=metrics,
            decoders=decoders
        )

###############################################################################
# Model Dictionary
###############################################################################

model_dict = {
    "Classifier": Classifier
}