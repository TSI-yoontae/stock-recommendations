import numpy as np

def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    return (x - np.nanmean(x)) / sd


def row_zscore(scores, available):
    out = np.full_like(scores, -np.inf, dtype=float)
    for u in range(scores.shape[0]):
        out[u, available[u]] = zscore(scores[u, available[u]])
    return out


def topk(scores, k):
    return [np.argsort(-scores[u])[:k] for u in range(scores.shape[0])]


def trmv_scores(ctx, alpha):
    scores = np.full_like(ctx['zwm'], -np.inf, dtype=float)
    available = np.isfinite(ctx['zwm']) & np.isfinite(ctx['zmvecf'])
    scores[available] = (1 - alpha) * ctx['zwm'][available] + alpha * ctx['zmvecf'][available]
    return scores


def equal_weights(n_users, k):
    return np.full((n_users, k), 1 / k, dtype=float)


def capped_kl_tilt(base, score, temperature, upper):
    base = np.asarray(base, dtype=float)
    base = base / base.sum()
    score = np.asarray(score, dtype=float)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), base.shape).copy()
    if temperature <= 0 or np.nanstd(score) < 1e-12:
        raw = base.copy()
    else:
        logits = np.log(np.maximum(base, 1e-300)) + temperature * zscore(score)
        logits -= np.max(logits)
        raw = np.exp(logits)
        raw /= raw.sum()
    if np.all(raw <= upper + 1e-12):
        return raw
    if upper.sum() < 1 - 1e-10:
        raise ValueError('infeasible upper bounds')
    out = np.zeros_like(raw)
    active = np.ones(len(raw), dtype=bool)
    remaining = 1.0
    while np.any(active):
        idx = np.flatnonzero(active)
        mass = raw[idx].sum()
        if mass <= 1e-300:
            scaled = np.full(len(idx), remaining / len(idx))
        else:
            scaled = raw[idx] / mass * remaining
        over = scaled > upper[idx] + 1e-12
        if not np.any(over):
            out[idx] = scaled
            break
        fixed = idx[over]
        out[fixed] = upper[fixed]
        remaining -= upper[fixed].sum()
        active[fixed] = False
    out /= out.sum()
    return out


def marginal_utility_weights(ctx, rec_lists, temperature, mean_shrink, cap):
    k = ctx['k']
    base = np.full(k, 1 / k)
    mu = ctx['mu_tr']
    sigma = ctx['Sigma_tr']
    w = ctx['W']
    gamma = ctx['gamma']
    delta = ctx['delta']
    mu_tilde = np.mean(mu) + mean_shrink * (mu - np.mean(mu))
    qmat = np.empty((ctx['n_users'], k), dtype=float)
    scores = np.empty((ctx['n_users'], k), dtype=float)
    for u, rec0 in enumerate(rec_lists):
        rec = np.asarray(rec0, dtype=int)
        s_rr = sigma[np.ix_(rec, rec)]
        cov_current = w[u] @ sigma[:, rec]
        risk = (1 - delta) * cov_current + delta * (s_rr @ base)
        marginal = mu_tilde[rec] - gamma[u] * risk
        scores[u] = marginal
        qmat[u] = capped_kl_tilt(base, marginal, temperature, np.full(k, cap))
    return qmat, scores


def self_financing_weights(ctx, rec_lists, temperature, args, tilt_buy=True, tilt_sell=True):
    n_users = ctx['n_users']
    k = ctx['k']
    delta = ctx['delta']
    sigma = ctx['Sigma_tr']
    mu = ctx['mu_tr']
    mu_tilde = np.mean(mu) + args.mean_shrink * (mu - np.mean(mu))
    qmat = np.zeros((n_users, k), dtype=float)
    sells = []
    buy_scores = np.zeros((n_users, k), dtype=float)
    sell_scores = []
    for u, rec0 in enumerate(rec_lists):
        rec = np.asarray(rec0, dtype=int)
        held = np.flatnonzero(ctx['W'][u] > 1e-15)
        buy_base = np.full(k, 1 / k)
        sell_base = ctx['W'][u, held].copy()
        sell_base /= sell_base.sum()
        provisional = (1 - delta) * ctx['W'][u].copy()
        provisional[rec] += delta * buy_base
        marginal = mu_tilde - ctx['gamma'][u] * (sigma @ provisional)
        bscore = marginal[rec]
        sscore = -marginal[held]
        if tilt_buy:
            q = capped_kl_tilt(buy_base, bscore, temperature, np.full(k, args.buy_cap))
        else:
            q = buy_base
        sell_upper = np.minimum(args.sell_cap, ctx['W'][u, held] / delta)
        if tilt_sell:
            s = capped_kl_tilt(sell_base, sscore, temperature, sell_upper)
        else:
            s = sell_base
        qmat[u] = q
        sells.append((held, s))
        buy_scores[u] = bscore
        sell_scores.append(sscore)
    return qmat, sells, buy_scores, sell_scores


def target_weighted(ctx, u, rec, q):
    target = (1 - ctx['delta']) * ctx['W'][u].copy()
    target[np.asarray(rec, dtype=int)] += ctx['delta'] * np.asarray(q, dtype=float)
    return target


def target_self_financing(ctx, u, rec, q, sell):
    held, s = sell
    target = ctx['W'][u].copy()
    target[np.asarray(held, dtype=int)] -= ctx['delta'] * np.asarray(s, dtype=float)
    target[np.asarray(rec, dtype=int)] += ctx['delta'] * np.asarray(q, dtype=float)
    if target.min() < -1e-10:
        raise RuntimeError('negative target weight')
    target[target < 0] = 0
    target /= target.sum()
    return target
