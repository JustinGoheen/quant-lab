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

import numpy as np
import quantstats as qs
import talib as ta


def log_returns(series):
    """computes log returns of a price series"""
    return np.log(series / series.shift(1))


def average_true_range(dataframe, high="high", low: str = "low", close: str = "close", period=14):
    """calculates the ATR of the price series"""
    return ta.ATR(dataframe[high], dataframe[low], dataframe[close], timeperiod=period)


def normalized_average_true_range(dataframe, high="high", low: str = "low", close: str = "close", period=14):
    """calculates the normalized ATR of the price series"""
    return ta.NATR(dataframe[high], dataframe[low], dataframe[close], timeperiod=period)


def aroon_oscillator(dataframe, high="high", low: str = "low", period=25):
    """calculates the AROON oscilator of a time series"""
    return ta.AROONOSC(dataframe[high], dataframe[low], timeperiod=20)


def relative_strength_index(series, period=21):
    return ta.RSI(series, timeperiod=period)


def moving_average(series, period=20):
    """calculates a simple moving average of the price series"""
    return series.rolling(period).mean()


def rate_of_change(series, period=20):
    return ta.ROC(series, timeperiod=period)


def expanding_rank(series):
    """forms the expanding percentile rank of the input vector"""
    return series.expanding().rank(pct=True)


def rolling_rank(series, period=252):
    """forms the rolling percentile rank of the input vector"""
    return series.rolling(period).rank(pct=True)


def strategy_metrics(returns):
    data = returns.copy()
    metrics = qs.reports.metrics(data, display=False).T
    metrics.rename(columns={"CAGR﹪": "CAGR", "Sortino/√2": "Adjusted Sortino"}, inplace=True)
    metrics = metrics.to_dict()
    metrics = {k: metrics[k]["Strategy"] for k in metrics}
    return metrics
