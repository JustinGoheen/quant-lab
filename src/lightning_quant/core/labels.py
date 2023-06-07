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
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from rich.progress import Progress


def find_results(resultspath):
    resultspath = os.path.join(os.getcwd(), resultspath)
    results = [i for i in os.listdir(resultspath) if i.startswith("best")]
    if results:
        return os.path.join(resultspath, results[0])
    else:
        raise Exception("no reults found")


def read_results(resultspath):
    resultspath = find_results(resultspath)
    with open(resultspath, "r") as file:
        results = json.load(file)
    return results


class LabelMaker:
    def __init__(
        self,
        rawdir: str = "data/raw/",
        cfgdir: str = "data/brute_results",
        labeldir: str = "data/labels",
        close_col: str = "close",
        timezone: str = "US/Eastern",
    ):
        results = read_results(cfgdir)
        self.data = pd.read_parquet(rawdir)
        self.data.reset_index(inplace=True)
        self.data.set_index("timestamp", inplace=True)
        self.close = close_col
        self.fast = results["Fast"]
        self.slow = results["Slow"]
        self.symbol = self.data["symbol"].iloc[0]
        self.timezone = timezone
        self.labeldir = labeldir

    def run(self):
        with Progress() as progress:
            task = progress.add_task("GENERATING LABELS", total=100)
            self.data["fast"] = self.data[self.close].rolling(self.fast).mean()
            self.data["slow"] = self.data[self.close].rolling(self.slow).mean()
            self.data.dropna(inplace=True)
            self.data["position"] = self.data["fast"] >= self.data["slow"]

            labels = self.data[["position"]]

            dt = str(datetime.now().astimezone(tz=ZoneInfo(self.timezone))).replace(" ", "_")
            fname = os.path.join(os.getcwd(), self.labeldir, f"{self.symbol.upper()}_{dt}.pq")
            labels.to_parquet(fname)

            while not progress.finished:
                progress.update(task, advance=1)
