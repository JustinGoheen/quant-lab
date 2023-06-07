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

import json
import os

import pandas as pd


def find_results(resultspath):
    results = [i for i in os.listdir(resultspath) if i.startswith("best")]
    if results:
        return os.path.join(dir, results)
    else:
        raise Exception("no reults found")
    return


def read_results(resultspath):
    resultspath = find_results(resultspath)
    with open(resultspath, "r") as file:
        results = json.load(file)
    return results


class LabelMaker:
    def __init__(self, rawdir: str = "data/raw/", cfgdir: str = "data/brute_results", close_col: str = "close"):
        results = read_results(cfgdir)
        self.data = pd.read_parquet(rawdir)
        self.close = close_col
        self.fast = results["Fast"]
        self.slow = results["Slow"]

    def run(self):
        self.data["fast"] = self.data[self.close].rolling(self)
