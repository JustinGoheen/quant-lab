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
from lightning_quant.data.features import FeatureEngineer
from lightning_quant.data.fetch import FetchBars
from lightning_quant.data.labels import LabelEngineer


class QuantAgent:
    def __init__(self, key: str, secret: str, symbols: Union[str, List]) -> None:
        self.key = key
        self.secret = secret
        self.symbols = symbols

    def run(self):
        fetchbars = FetchBars(self.key, self.secret)
        fetchbars.run(symbol_or_symbols=self.symbols)
        FeatureEngineer().run()
        BruteForceOptimizer().run()
        LabelEngineer().run()
