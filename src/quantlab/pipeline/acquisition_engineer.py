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

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from rich.progress import Progress


class AcquisitionEngineer:
    def acquire_data(
        self,
        **kwargs,
    ):
        """
        Notes:
            StockBarsRequest:
            - https://alpaca.markets/docs/python-sdk/api_reference/data/stock/requests.html#stockbarsrequest
        """
        client = StockHistoricalDataClient(self.api_key, self.api_secret)

        five_years_ago_today = datetime.datetime.now() - datetime.timedelta(days=5 * 365)

        request = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Day,
            start=five_years_ago_today,
            **kwargs,
        )

        with Progress() as progress:
            task = progress.add_task("Fetching Bars...", total=100)
            data = client.get_stock_bars(request)

            while not progress.finished:
                progress.update(task, advance=1)

            os.mkdir(self.agentdatapath)
            self.rawdatapath = os.path.join(self.agentdatapath, "raw_market_data_.pq")
            data.df.to_parquet(self.rawdatapath)
