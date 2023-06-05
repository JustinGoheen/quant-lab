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
from typing import List, Optional, Union

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from rich.progress import Progress


class FetchBars:
    def __init__(self, api_key, secret_key):
        self.client = StockHistoricalDataClient(api_key, secret_key)

    def run(
        self,
        symbol_or_symbols: Union[str, List],
        datadir: Optional[str] = "data",
        **kwargs,
    ):
        """
        Notes:
            StockBarsRequest: https://alpaca.markets/docs/python-sdk/api_reference/data/stock/requests.html#stockbarsrequest
        """
        five_years_ago_today = datetime.datetime.now() - datetime.timedelta(days=5 * 365)

        request = StockBarsRequest(
            symbol_or_symbols=symbol_or_symbols,
            timeframe=TimeFrame.Day,
            start=five_years_ago_today,
            **kwargs,
        )

        with Progress() as progress:
            task = progress.add_task("Fetching Bars...", total=100)
            data = self.client.get_stock_bars(request)

            while not progress.finished:
                progress.update(task, advance=0.1)

            datapath = os.path.join(os.getcwd(), datadir, f"market_data_{str(datetime.datetime.now())}.pq")
            data.df.to_parquet(datapath)
