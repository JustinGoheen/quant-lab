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

import click

from lightning_quant.core.brute import BruteForceOptimizer
from lightning_quant.core.features import FeatureEngineer
from lightning_quant.core.fetch import FetchBars


@click.group()
def main() -> None:
    pass


@main.command("fetch-data")
@click.option("--key")
@click.option("--secret")
@click.option("--symbols", default="SPY")
def fetch_data(key, secret, symbols) -> None:
    app = FetchBars(key, secret)
    app.run(symbol_or_symbols=symbols)


@main.command("make-features")
def make_features() -> None:
    app = FeatureEngineer()
    app.run()


@main.command("run-bfo")
def run_bfo() -> None:
    app = BruteForceOptimizer()
    app.run()
