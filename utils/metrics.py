import math

import numpy as np


def max_drawdown(monthly_returns):
    wealth = np.cumprod(1 + np.asarray(monthly_returns, dtype=float))
    path = np.r_[1.0, wealth]
    peak = np.maximum.accumulate(path)
    return float(np.max(1 - path / peak))


def cvar_loss(monthly_returns, alpha=0.05):
    returns = np.sort(np.asarray(monthly_returns, dtype=float))
    n_tail = max(1, int(math.ceil(alpha * len(returns))))
    return float(-np.mean(returns[:n_tail]))


def sortino_ratio(monthly_returns):
    returns = np.asarray(monthly_returns, dtype=float)
    downside = np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * math.sqrt(12)
    annual_mean = float(np.mean(returns) * 12)
    return annual_mean / max(downside, 1e-12)


def path_statistics(monthly_returns, gamma):
    returns = np.asarray(monthly_returns, dtype=float)
    annual_mean = float(np.mean(returns) * 12)
    annual_var = float(np.var(returns, ddof=1) * 12)
    annual_vol = math.sqrt(max(annual_var, 1e-12))
    return {
        'mean': annual_mean,
        'var': annual_var,
        'vol': annual_vol,
        'ce': annual_mean - 0.5 * gamma * annual_var,
        'sr': annual_mean / annual_vol,
        'sortino': sortino_ratio(returns),
        'mdd': max_drawdown(returns),
        'cvar': cvar_loss(returns),
        'terminal_wealth': float(np.prod(1 + returns)),
    }


def average_precision_at_k(rec, relevant, k):
    hit = np.isin(rec[:k], relevant).astype(float)
    if not hit.any():
        return 0.0
    precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
    return float(np.sum(precision * hit) / max(1, min(k, len(relevant))))
