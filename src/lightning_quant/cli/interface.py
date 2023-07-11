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

import typer
from dotenv import load_dotenv
from typing_extensions import Annotated

from lightning_quant.core.agent import QuantAgent
from lightning_quant.core.lightning_trainer import QuantLightningTrainer
from lightning_quant.data.datamodule import MarketDataModule
from lightning_quant.models.elasticnet import ElasticNet
from lightning_quant.models.mlp import ElasticNetMLP

load_dotenv()

# the main app
# use this in setup.cfg in options.entry_points
app = typer.Typer()
# a nested app called "run"
run_app = typer.Typer()
app.add_typer(run_app, name="run")


@run_app.callback()
def run_callback():
    pass


@run_app.command("agent")
def agent(
    key: Annotated[str, typer.Option()] = os.environ["API_KEY"],
    secret: Annotated[str, typer.Option()] = os.environ["SECRET_KEY"],
    symbol: Annotated[str, typer.Option()] = "SPY",
    tasks: Annotated[str, typer.Option()] = "all",
) -> None:
    tasks = [i.replace("=", "") for i in tasks]
    if len(tasks) == 1:
        tasks = tasks[0]
    agent = QuantAgent(api_key=key, api_secret=secret, symbol=symbol)
    agent.run(tasks)


@run_app.command("fast-dev")
def fast_dev(
    model: Annotated[str, typer.Option(help="a model name from (elasticnet, mlp)")] = "elasticnet",
    num_classes: Annotated[int, typer.Option(help="the number of classes or labels")] = 2,
    accelerator: Annotated[str, typer.Option(help="one of (cpu, gpu, tpu, ipu, auto)")] = "cpu",
    devices: Annotated[int, typer.Option()] = 1,
    strategy: Annotated[str, typer.Option()] = None,
) -> None:
    print("running typer")
    models = {"elasticnet": ElasticNet, "mlp": ElasticNetMLP}
    model = models[model]
    model = model(in_features=6, num_classes=num_classes)
    datamodule = MarketDataModule()
    if strategy:
        trainer = QuantLightningTrainer(fast_dev_run=True, devices=devices, accelerator=accelerator, strategy=strategy)
    else:
        trainer = QuantLightningTrainer(fast_dev_run=True, devices=devices, accelerator=accelerator)
    trainer.fit(model=model, datamodule=datamodule)
