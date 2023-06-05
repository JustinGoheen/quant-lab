# Feature Engineering

## The Basics

Features in machine learning are simply the columns in a dataframe or excel sheet e.g. the raw features after acquiring the data from IBKR would be the open, high, low, and close columns. We would need to create a target (y) from the close feature by creating a new column, log_returns as:

$$
log\text{-}returns = ln(\frac{close_{t}}{close_{t+1}})
$$

where $ln$ is the natural log, and $\frac{close_{t}}{close_{t+1}}$ is today's close divided by tomorrow's close.

Using Pandas and NumPy, this would look like:

```python
df = pd.read_parquet("../data", columns=["open", "high", "low", "close"])

df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
```

## Technical Indicators as Features

Disclaimer: Arbitrary use of technical indicators is not recommended. There should be some sense of why a particular indicator is being used.

[Ta-Lib](https://github.com/ta-lib/ta-lib-python) (Cython) is the recommended library for technical indicators. The library depends on the SWIG version [https://ta-lib.org] and that version _must_ be installed. You can read more in the following link https://github.com/ta-lib/ta-lib-python#dependencies.

Ta-Lib docs do not explain indicators. It will be neccessary to research what an indicators is, and why it might be used.
