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
from typing import List, Union
from zoneinfo import ZoneInfo

from lightning_quant.pipeline.acquisition_engineer import AcquisitionEngineer
from lightning_quant.pipeline.brute import BruteForceOptimizer
from lightning_quant.pipeline.feature_engineer import FeatureEngineer
from lightning_quant.pipeline.label_engineer import LabelEngineer


class QuantAgent(FeatureEngineer, AcquisitionEngineer, LabelEngineer, BruteForceOptimizer):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: Union[str, List],
        datadir="data",
        timezone: str = "US/Eastern",
        cagr: float = 0.5,
        sharpe=0.5,
        max_drawdown=-0.3,
        close_col: str = "close",
        columns: List[str] = ["open", "high", "low", "close", "vwap", "symbol", "timestamp"],
    ) -> None:
        """runs agent pipeline for data acquisition and feature engineering"""

        if not isinstance(symbol, str):
            raise Exception(f"please pass a single market symbol as a str. got {type(symbol)}")

        self.api_key = api_key
        self.api_secret = api_secret
        self.datadir = datadir
        self.timezone = timezone
        self.symbol = symbol
        self.cagr = cagr
        self.sharpe = sharpe
        self.max_drawdown = max_drawdown
        self.close_col = close_col
        self.columns = columns

        dt = str(datetime.now().astimezone(tz=ZoneInfo(self.timezone))).replace(" ", "_")
        self.agentdatapath = os.path.join(os.getcwd(), self.datadir, f"{self.symbol.upper()}_{dt}")
        self.rawdatapath = None
        self.resultspath = None
        self.bestconfigpath = None
        self.labelspath = None
        self.featurespath = None

        super().__init__()

    def run(self, tasks: Union[str, List[str]] = "all"):
        """
        # Args
            tasks: a str or list of strs of any combination of ()
        """
        allowed_tasks = ["acquire", "optimize", "features", "labels"]
        if isinstance(tasks, list):
            if "all" in tasks:
                raise Exception(
                    "please pass `all` or some combination of (`acquire`, `optimize`, `features`, `labels`)"
                )
            if not all(i in allowed_tasks for i in tasks):
                not_allowed = "".join([i for i in tasks if i not in allowed_tasks])
                raise Exception(f"{not_allowed} is not a valid task")

        if "acquire" in tasks or tasks == "all":
            self.acquire_data()
        if "optimize" in tasks or tasks == "all":
            self.optimize_moving_averages()
        if "features" in tasks or tasks == "all":
            self.engineer_features()
        if "labels" in tasks or tasks == "all":
            self.engineer_labels()
