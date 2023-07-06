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

import datetime
import os

from pyarrow import parquet as pq
from rich import print as rprint
from rich.progress import Progress

from lightning_quant.factors.ta import (
    aroon_oscillator,
    expanding_rank,
    log_returns,
    normalized_average_true_range,
    rate_of_change,
    relative_strength_index,
)


class FeatureEngineer:
    def _locate_raw_data(self):
        [i for i in os.listdir(self.rawdatapath) if i.startswith("")]

    def engineer_features(
        self,
        timezone: str = "US/Eastern",
    ):
        """
        a pipeline to create technical analysis features

        Notes:
            - when appropriate, the features will be normalized and created as expanding windows
            - indicators are: ATR, aroon oscillator, RSI, rate of change,
        """
        rprint(f"[{datetime.datetime.now().time()}] STARTING PRE PROCESSING")

        self.data = pq.read_table(self.rawdatapath, columns=self.columns).to_pandas()
        self.data.reset_index(inplace=True)
        self.data.set_index("timestamp", inplace=True)
        # set market symbols
        self.symbol = self.data["symbol"].unique().tolist()
        # set unlearned moving average lookback windows
        # these can be learned later through brute force optimization
        # bfo is simply a for loop that selects the best cfg on some
        # heuristic like minimum max-drawdown or maximum CAGR
        self.fast_window = 20
        self.slow_window = 50

        with Progress() as progress:
            task = progress.add_task("PROCESSING DATA", total=100)

            for symbol in self.symbol:
                # filter data to symbol
                batch = self.data.loc[self.data["symbol"] == symbol, :]

                # normalized average true range
                batch[f"{symbol.upper()}_NATR_RANK"] = expanding_rank(
                    normalized_average_true_range(batch, period=self.slow_window)
                )
                # aroon oscillator
                batch[f"{symbol.upper()}_AROON_RANK"] = expanding_rank(aroon_oscillator(batch, period=self.slow_window))
                # RSI
                batch[f"{symbol.upper()}_RSI_RANK"] = expanding_rank(
                    relative_strength_index(batch["close"], period=self.slow_window)
                )
                # ROC
                batch[f"{symbol.upper()}_ROC_RANK"] = expanding_rank(
                    rate_of_change(batch["close"], period=self.slow_window)
                )
                # log returns
                batch[f"{symbol.upper()}_RTNS_RANK"] = expanding_rank(log_returns(batch["close"]))
                # expanding rank vwap
                if "vwap" in batch.columns:
                    batch[f"{symbol.upper()}_VWAP_RANK"] = expanding_rank(batch["vwap"])

                batch.drop(["open", "high", "low", "close"], axis=1, inplace=True)
                batch.dropna(inplace=True)
                batch.index = batch.index.date

                featurespath = os.path.join(self.agentdatapath, "features.pq")
                # drop null values and save
                batch.dropna().to_parquet(featurespath)

            while not progress.finished:
                progress.update(task, advance=1)

        rprint(f"[{datetime.datetime.now().time()}] PROCESSING COMPLETE")
