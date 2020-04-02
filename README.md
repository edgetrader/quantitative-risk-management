# Efficient Frontier Portfolio Optimisation for Strategic Asset Allocation

## Summary
This project attempts to find the optimal portfolio weights of selected assets using efficient frontier portfolio optimisation methodology.

It also provides a performance and risk analysis summary on the optimal portfolio (with Max Sharpe Ratio) using the Pyfolio library.

## Data
Data downloaded using yfinance python library

## Efficient Frontier Model
### PyPortfolioOpt
PyPortfolioOpt is a library that implements portfolio optimisation methods, including classical efficient frontier techniques and Black-Litterman allocation, as well as more recent developments in the field like shrinkage and Hierarchical Risk Parity, along with some novel experimental features like exponentially-weighted covariance matrices.

https://pyportfolioopt.readthedocs.io/en/latest/


## Performance and Risk Analysis of Optimal Portfolio
### Pyfolio
pyfolio is a Python library for performance and risk analysis of financial portfolios developed by Quantopian Inc. It works well with the Zipline open source backtesting library. Quantopian also offers a fully managed service for professionals that includes Zipline, Alphalens, Pyfolio, FactSet data, and more.

https://github.com/quantopian/pyfolio


## Take a look at the report
Clone the repo and run the jupyter notebook inside ./notebook/asset-allocation.ipynb
or access the link here: https://github.com/edgetrader/asset-allocation/blob/master/notebook/asset-allocation.ipynb
