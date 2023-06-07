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

import os
from datetime import datetime
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from rich import print as rprint
from rich.progress import Progress

from lightning_quant.core.factors import log_returns, strategy_metrics


class BruteForceOptimizer:
    """optimizes for a dual moving average given a selection heuristic


    Note:
        see page 490 of Dr. yves Hilpicsh's Python for Finance second edition
    """

    def __init__(
        self,
        cagr=0.5,
        sharpe=0.5,
        max_drawdown=-0.3,
        price_column: str = "close",
        rawdatadir: str = "data/raw",
        resultsdir: str = "data/brute_results",
    ):
        self.cagr = cagr
        self.sharpe = sharpe
        self.max_drawdown = -max_drawdown if max_drawdown > 0 else max_drawdown
        self.price_column = price_column
        self.resultsdir = resultsdir

        self.results = pd.DataFrame(columns=["Fast", "Slow", "CAGR", "Sharpe", "Drawdown", "Returns"])

        rawpath = os.path.join(os.getcwd(), rawdatadir)
        self.rawdata = pd.read_parquet(rawpath, columns=[self.price_column])
        self.rawdata.reset_index(inplace=True)
        self.rawdata.set_index("timestamp", inplace=True)
        self.rawdata.drop("symbol", axis=1)
        self.rawdata.index = pd.to_datetime(self.rawdata.index.date)
        self.rawdata["returns"] = log_returns(self.rawdata[self.price_column])
        self.rawdata.dropna(inplace=True)

    def run(
        self,
        timezone: str = "US/Eastern",
    ):
        rprint(f"[{datetime.now().time()}] STARTING BFO")

        fast_range = range(10, 51, 1)
        slow_range = range(50, 125, 1)

        with Progress() as progress:
            task = progress.add_task("BRUTE FORCE OPTIMIZATION", total=len(fast_range) * len(slow_range))

            for fast, slow in product(fast_range, slow_range):
                testdata = self.rawdata.copy()
                if fast != slow:  # account for 50, 50 overlap
                    testdata["fast"] = testdata[self.price_column].rolling(fast).mean()
                    testdata["slow"] = testdata[self.price_column].rolling(slow).mean()
                    testdata["position"] = np.where(testdata["fast"] >= testdata["slow"], 1, 0)
                    testdata["strategy_returns"] = testdata["position"] * testdata["returns"]  # do not shift position
                    testdata.dropna(inplace=True)
                    metrics = strategy_metrics(testdata["strategy_returns"])
                    payload = {
                        "Fast": fast,
                        "Slow": slow,
                        "CAGR": metrics["CAGR"],
                        "Sharpe": metrics["Sharpe"],
                        "Drawdown": metrics["Max Drawdown"],
                        "Returns": np.exp(testdata["strategy_returns"].sum()),
                    }
                    self.results = pd.concat(
                        [
                            self.results,
                            pd.DataFrame(payload),
                        ],
                    )
                    rprint(
                        f"[{datetime.now().time()}]: Fast: {fast} Slow: {slow} CAGR: {metrics['CAGR'].iloc[0]} DD: {metrics['Max Drawdown'].iloc[0]}"  # noqa: E501
                    )

                    progress.advance(task)

        self.results = self.results.loc[self.results["Drawdown"] >= self.max_drawdown, :]
        self.results.sort_values("Returns", ascending=False, inplace=True)

        dt = str(datetime.now().astimezone(tz=ZoneInfo(timezone))).replace(" ", "_")
        resultsfname = os.path.join(self.resultsdir, f"results_{dt}.pq")
        self.results.to_csv(resultsfname)

        best = self.results.iloc[0]
        best = pd.DataFrame(best).T
        bestfname = os.path.join(self.resultsdir, f"best_{dt}.pq")
        best.to_parquet(bestfname)

        rprint(
            f"[{datetime.now().time()}] BFO RESULTS: CAGR {best['CAGR'].iloc[0]}, DD {best['Drawdown'].iloc[0]}, Fast {best['Fast'].iloc[0]}, Slow {best['Slow'].iloc[0]}"  # noqa: E501
        )
