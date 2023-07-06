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

import click
from dotenv import load_dotenv

from lightning_quant.core.agent import QuantAgent

load_dotenv()


@click.group()
def main() -> None:
    pass


@main.group("run")
def run():
    pass


@run.command("agent")
@click.option("--key", default=os.environ["API_KEY"])
@click.option("--secret", default=os.environ["SECRET_KEY"])
@click.option("--symbol", default="SPY")
@click.option("--tasks", "-t", multiple=True)
def agent(key, secret, symbol, tasks):
    tasks = [i.replace("=", "") for i in tasks]
    if len(tasks) == 1:
        tasks = tasks[0]
    agent = QuantAgent(api_key=key, api_secret=secret, symbol=symbol)
    agent.run(tasks)
