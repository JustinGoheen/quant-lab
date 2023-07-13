# Copyright Justin R. Goheen.
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


def regularization(model, loss, l1_strength=0.5, l2_strength=0.5):
    output_layer = -1
    if l1_strength > 0:
        l1_reg = model.sequential[output_layer].weight.abs().sum()
        loss += l1_strength * l1_reg
    if l2_strength > 0:
        l2_reg = model.sequential[output_layer].weight.pow(2).sum()
        loss += l2_strength * l2_reg
    return loss
