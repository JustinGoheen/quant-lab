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
from itertools import product

import numpy as np
import pandas as pd
from rich import print as rprint
from rich.progress import Progress

from lightning_quant.factors.ta import log_returns, strategy_metrics


class BruteForceOptimizer:
    """optimizes for a dual moving average given a selection heuristic


    Note:
        see page 490 of Dr. yves Hilpicsh's Python for Finance second edition
    """

    def optimize_moving_averages(
        self,
        timezone: str = "US/Eastern",
    ):
        rprint(f"[{datetime.now().time()}] STARTING BFO")

        data = pd.read_parquet(self.rawdatapath, columns=[self.close_col])
        data.reset_index(inplace=True)
        data.set_index("timestamp", inplace=True)
        data.drop("symbol", axis=1)
        data.index = pd.to_datetime(data.index.date)
        data["returns"] = log_returns(data[self.close_col])
        data.dropna(inplace=True)

        results = []

        fast_range = range(10, 51, 1)
        slow_range = range(50, 125, 1)

        with Progress() as progress:
            task = progress.add_task("BRUTE FORCE OPTIMIZATION", total=len(fast_range) * len(slow_range))

            sentinel = 0
            for fast, slow in product(fast_range, slow_range):
                testdata = data.copy()
                if fast != slow:  # account for 50, 50 overlap
                    testdata["fast"] = testdata[self.close_col].rolling(fast).mean()
                    testdata["slow"] = testdata[self.close_col].rolling(slow).mean()
                    testdata["position"] = np.where(testdata["fast"] >= testdata["slow"], 1, 0)
                    testdata["strategy_returns"] = testdata["position"] * testdata["returns"]  # do not shift position
                    testdata.dropna(inplace=True)
                    metrics = strategy_metrics(testdata["strategy_returns"])

                    payload = {
                        "Trial": sentinel,
                        "Fast": fast,
                        "Slow": slow,
                        "CAGR": metrics["CAGR"],
                        "Sharpe": metrics["Sharpe"],
                        "Drawdown": metrics["Max Drawdown"],
                        "Returns": np.exp(testdata["strategy_returns"].sum()),
                    }

                    results.append(payload)

                    rprint(
                        f"[{datetime.now().time()}]: Fast: {fast} Slow: {slow} CAGR: {metrics['CAGR']} DD: {metrics['Max Drawdown']}"  # noqa: E501
                    )

                    if sentinel == 0:
                        best_returns = payload["Returns"]
                        best = payload
                    else:
                        if payload["Returns"] > best_returns:
                            best_returns = payload["Returns"]
                            best = payload

                    sentinel += 1

                    progress.advance(task)

        results = pd.DataFrame(results).set_index("Trial")
        results = results.loc[results["Drawdown"] >= self.max_drawdown, :]
        results = results.loc[results["Returns"] >= 1.0, :]
        results.sort_values("Returns", ascending=False, inplace=True)

        self.resultspath = os.path.join(self.agentdatapath, "results.csv")
        results.to_csv(self.resultspath)

        self.bestconfigpath = os.path.join(self.agentdatapath, "best_config.json")

        with open(self.bestconfigpath, "w") as bestcfg:
            json.dump(best, bestcfg, indent=4)

        rprint(
            f"[{datetime.now().time()}] BFO RESULTS: CAGR {best['CAGR']}, DD {best['Drawdown']}, Fast {best['Fast']}, Slow {best['Slow']}"  # noqa: E501
        )
