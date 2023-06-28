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

from typing import List, Union

from lightning_quant.core.brute import BruteForceOptimizer
from lightning_quant.core.features import FeatureEngineer
from lightning_quant.core.fetch import FetchBars
from lightning_quant.core.labels import LabelEngineer


class QuantAgent:
    def __init__(self, key: str, secret: str, symbols: Union[str, List]) -> None:
        self.fetchbars = FetchBars(key, secret)
        self.featureengineer = FeatureEngineer()
        self.bfo = BruteForceOptimizer()
        self.labelengineer = LabelEngineer()
        self.symbols = symbols

    def run(self):
        self.fetchbars.run(symbol_or_symbols=self.symbols)
        self.featureengineer.run()
        self.bfo.run()
        self.labelengineer.run()
