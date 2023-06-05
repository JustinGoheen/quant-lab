# Lightning Quant

<!-- # Copyright Justin R. Goheen.
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
# limitations under the License. -->

Trading agents built with [Lightning AI](https://lightning.ai/)'s Lightning Fabric and TorchMetrics, Nixtla's [neuralforecast](https://github.com/Nixtla/neuralforecast), and Alpaca Markets' [alpaca-py](https://alpaca.markets/docs/python-sdk/).

[SPY](https://www.google.com/finance/quote/SPY:NYSEARCA?sa=X&ved=2ahUKEwjQ-MKp5az_AhV2mYQIHXfxCu4Q3ecFegQIJRAX) (S&P 500) is used in examples.

## High Level Roadmap

- [x] fetch data
- [ ] feature engineering
- [ ] model selection
- [ ] hyperparameter optimization
- [ ] train
- [ ] write trading agent

## Preparing to use Lightning-Quant

Fork then clone the repo. After cloning the repo to your machine, do the following in terminal:

```bash
cd {{ path to clone }}
python3 -m venv .venv/
source .venv/bin/activate
pip install -e .
```

## Requirements

The instructions shown above will install the base requirements, those requirements are:

- PyTorch
- [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
- [Weights and Biases](https://docs.wandb.ai/guides): Experiment Manager
- [alpaca-py](https://alpaca.markets/docs/python-sdk/): Alpaca Markets Python API
- [neuralforecast](https://github.com/Nixtla/neuralforecast): neural forecasting models focused by Nixtla
- [Ploty](https://plotly.com/python/): Data Visualization
- [Typer](https://typer.tiangolo.com): Command Line Interfaces

## Additional Requirements

To install the Cython version of TA-Lib, you must first install the SWIG version. See [the notes](https://github.com/ta-lib/ta-lib-python#dependencies) in the Cython version for installation instructions.

## Additional Resources

### PyTorch

The books shown below will be used as references during the machine learning phase of this project.

1. Deep Learning with PyTorch, Stevens et al
2. Machine Learning with PyTorch and Scikit-Learn, Raschka et al

### Algorithmic Trading

Each of the books below are by Dr. Yves Hilpisch. This series of referencess will drive examples.

> The neural networks found in the texts use TensorFlow

1. Financial Theory with Python
2. Python for Algorithmic Trading
3. Artificial Intelligence in Finance
