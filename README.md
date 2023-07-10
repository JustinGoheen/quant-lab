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

Lightning Quant is a library for training algorithmic trading agents with [Lightning AI](https://lightning.ai/) [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and [Lightning Fabric](https://lightning.ai/docs/fabric/stable/), along with the following ecosystem projects:

- [neuralforecast](https://github.com/Nixtla/neuralforecast)
- [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/)
- [SheepRL](https://github.com/Eclectic-Sheep/sheeprl)

Lightning AI's PyTorch Lightning and Lightning Fabric are agnostic to the market broker and data source. One needs only to acquire and preprocess the desired market data and then construct the requisite PyTorch DataLoaders and LightningDataModule for the PyTorch Lightning Trainer or Lightning Fabric training loop that will be used with the bespoke PyTorch model, a SheepRL algorithm, or a neuralforecast model.

[Alpaca Markets](https://alpaca.markets/) is used to fetch the historical data for the exercise.

[SPY](https://www.google.com/finance/quote/SPY:NYSEARCA?sa=X&ved=2ahUKEwjQ-MKp5az_AhV2mYQIHXfxCu4Q3ecFegQIJRAX) (S&P 500) is used in examples.

## Setup

First – fork, then clone the repo. After cloning the repo to your machine, do the following in terminal to navigate to your clone:

```sh
cd {{ path to clone }}
```

> **Note**
>
> SheepRL requires a Python version less than 3.11 and greater than or equal to 3.8.

If you have Python 3.10, 3.9, or 3.8 as system Python, you can create and activate a virtual environment with:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

If you have Python 3.11 or later as system Python, but have [homebrew](https://brew.sh/) installed, you can install Python 3.10 and then create the venv with:

```sh
brew install python@3.10
python3.10 -m venv .venv
source .venv/bin/activate
```

If you have Python 3.11 or later as system Python, but have [conda](https://docs.conda.io/en/latest/) or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed, create the conda env with:

```sh
conda create -n lit-quant python=3.10 -y
conda activate lit-quant
```

Then, install an editable version of lightning-quant with:

```sh
pip install -e .
```

> **Note**
>
> ensure your venv or conda env is activated before proceeding

> **Note**
>
> the example uses pip regardless of if you've created your env with venv or conda

> **Note**
>
> if you are on an Apple Silicon powered MacBook and encounter an error attributed box2dpy during install, you need to install SWIG using the instructions shown below to support gym and gymnasium.

It is recommended to use [homebrew](https://brew.sh/) to install [SWIG](https://formulae.brew.sh/formula/swig) to support [Gym](https://github.com/openai/gym).

```sh
# if needed, install homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# then, do
brew install swig
# then attempt to pip install again
pip install -e .
```

## Requirements

The instructions shown above will install the base requirements, those requirements are:

- PyTorch
- [Lightning Fabric](https://lightning.ai/docs/fabric/stable/)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
- [Weights and Biases](https://docs.wandb.ai/guides): Experiment Manager
- [alpaca-py](https://alpaca.markets/docs/python-sdk/): Alpaca Markets Python API
- [neuralforecast](https://github.com/Nixtla/neuralforecast): neural forecasting models created by Nixtla
- [Ploty](https://plotly.com/python/): Data Visualization
- [Click](https://click.palletsprojects.com/): Command Line Interfaces

## Additional Requirements

### Installing TA-LIB on MacOS

> **Note**
>
> To install the Cython version of TA-Lib, you must first install the SWIG version.

```sh
# if needed, install homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# then, do
brew install ta-lib
# then install Cython Ta-Lib
pip install TA-lib
```

### Installing TA-Lib on Linux

> **Note**
>
> You will need to be in your root user directory

> **Note**
>
> The instructions shown below uses the lightning-quant path as a custom PREFIX

> **Note**
>
> replace {YOUR_USERNAME} with your username

To install the source version in Linux, do the following:

```sh
# get the zipped package
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# unzip
tar -xzf ta-lib-0.4.0-src.tar.gz
# remove the zipped package
rm -rf http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
# create a custom prefix as the path to lightning-quant
export TALIB_PREFIX=$PWD/.venv/ctalib
# change directoy into the unzipped package
cd ta-lib/
# install
./configure --prefix=$TALIB_PREFIX
make && make install
# set env variables to locate the install
export TA_INCLUDE_PATH=$TALIB_PREFIX/include
export TA_LIBRARY_PATH=$TALIB_PREFIX/lib
```

Navigate back to root directory of lightning-quant and do the following:

```sh
# activate the venv and install
cd ..
# ensure you are in lightning-quant
pwd
# if in lightning quant, proceed
git clone https://github.com/TA-Lib/ta-lib-python.git
source .venv/bin/activate
pip install ta-lib-python/
```

## Using Lightning-Quant

Lightning-Quant provides a CLI, `quant` built with [Click]().

![](docs/assets/lightning-quant-run.png)

To run data acquisition, feature engineering, brute force optimization, and label generation at one time, do:

```sh
quant run agent --key-YOUR-ALPACA-KEY --secret=YOUR-ALPACA-SECRET-KEY --symbol=SPY
```

Alternatively, you can create a .env file and lightning-quant will automatically load the provided environment variables for you. And then use the following in terminal:

```sh
quant run agent --symbol=SPY --tasks=all
```

![](docs/assets/agent-run.gif)

> **Warning**
>
> do not commit your .env files to GitHub

The contents of your `.env` file should be:

```txt
API_KEY=YOUR_API_KEY
SECRET_KEY=YOUR_SECRET_KEY
```

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

The following book is on ML for factor investing. The implementations are written in R in the book, and the community has provided select Python implementations.

https://www.mlfactor.com/
