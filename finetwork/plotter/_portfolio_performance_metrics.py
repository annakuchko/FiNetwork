import numpy as np
import pandas as pd

def volatility(portfolio_returns):
    vol = portfolio_returns.std()
    return vol

def sharpe_ratio(portfolio_returns, risk_free=0):
    mean_adj = portfolio_returns.mean() * 255 - risk_free
    sigma = portfolio_returns.std() * np.sqrt(255)
    ratio = mean_adj / sigma
    return ratio

def sortino_ratio(portfolio_returns, risk_free=0):
    mean_adj = portfolio_returns.mean() * 255 - risk_free
    std_neg = portfolio_returns[portfolio_returns<0].std() * np.sqrt(255)
    ratio = mean_adj / std_neg
    return ratio

def max_drawdown(portfolio_returns):
    comp_ret = (portfolio_returns + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    max_dd = dd.min()
    return max_dd

def calmar_ratio(portfolio_returns):
    max_drawdowns = max_drawdown(portfolio_returns)
    calmars = portfolio_returns.mean() * 255 / abs(max_drawdowns)
    return calmars

if __name__ == '__main__':
    ret = [1.12, 1.14, 0.98, 0.67, 1.01, 0.998, 1.56, np.nan]
    ret.append(np.nan)
    print(ret)
    ret = pd.Series(ret)-1
    print(f'volatility: {volatility(ret)}')
    print(f'sharpe_ratio: {sharpe_ratio(ret)}')
    print(f'sortino_ratio: {sortino_ratio(ret)}')
    print(f'max_drawdown: {max_drawdown(ret)}')
    print(f'calmar_ratio: {calmar_ratio(ret)}')