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
from rich.progress import Progress


def read_results(resultspath):
    with open(resultspath, "r") as file:
        results = json.load(file)
    return results


class LabelEngineer:
    def engineer_labels(self):
        results = read_results(self.bestconfigpath)
        data = pd.read_parquet(self.rawdatapath)
        data.reset_index(inplace=True)
        data.set_index("timestamp", inplace=True)
        fast = results["Fast"]
        slow = results["Slow"]

        with Progress() as progress:
            task = progress.add_task("GENERATING LABELS", total=100)
            data["fast"] = data[self.close_col].rolling(fast).mean()
            data["slow"] = data[self.close_col].rolling(slow).mean()
            data.dropna(inplace=True)
            data["position"] = data["fast"] >= data["slow"]

            labels = data[["position"]]
            labels.index = labels.index.date

            fname = os.path.join(self.agentdatapath, "labels.pq")
            labels.to_parquet(fname)

            while not progress.finished:
                progress.update(task, advance=1)
