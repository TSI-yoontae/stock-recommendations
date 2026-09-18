import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from models.mvecf import mvecf_wmf
from models.trmv_sf import row_zscore, zscore
from models.wmf import implicit_als


def ff12_industry(sic):
    try:
        s = int(float(sic))
    except Exception:
        return 11
    if (100 <= s <= 999) or (2000 <= s <= 2399) or (2700 <= s <= 2749) or (2770 <= s <= 2799) or (3100 <= s <= 3199) or (3940 <= s <= 3989):
        return 0
    if (2500 <= s <= 2519) or (2590 <= s <= 2599) or (3630 <= s <= 3659) or s in (3710, 3711, 3714, 3716) or (3750 <= s <= 3751) or s == 3792 or (3900 <= s <= 3939) or (3990 <= s <= 3999):
        return 1
    if (2520 <= s <= 2589) or (2600 <= s <= 2699) or (2750 <= s <= 2769) or (3000 <= s <= 3099) or (3200 <= s <= 3569) or (3580 <= s <= 3629) or (3700 <= s <= 3709) or (3712 <= s <= 3713) or s in (3715, 3717) or (3720 <= s <= 3749) or (3752 <= s <= 3791) or (3793 <= s <= 3799) or (3830 <= s <= 3839) or (3860 <= s <= 3899):
        return 2
    if (1200 <= s <= 1399) or (2900 <= s <= 2999):
        return 3
    if (2800 <= s <= 2829) or (2840 <= s <= 2899):
        return 4
    if (3570 <= s <= 3579) or (3660 <= s <= 3692) or (3694 <= s <= 3699) or (3810 <= s <= 3829) or (7370 <= s <= 7379):
        return 5
    if 4800 <= s <= 4899:
        return 6
    if 4900 <= s <= 4949:
        return 7
    if (5000 <= s <= 5999) or (7200 <= s <= 7299) or (7600 <= s <= 7699):
        return 8
    if (2830 <= s <= 2839) or s == 3693 or (3840 <= s <= 3859) or (8000 <= s <= 8099):
        return 9
    if 6000 <= s <= 6999:
        return 10
    return 11


def moments(returns):
    mu = np.mean(returns, axis=0) * 12
    sigma = LedoitWolf().fit(returns).covariance_ * 12
    return mu, (sigma + sigma.T) / 2


def build_context(monthly, year, n_users, seed, args):
    rng = np.random.default_rng(seed + year)
    dec = pd.Timestamp(year, 12, 1)
    train_start = pd.Timestamp(year - args.lookback_years + 1, 1, 1)
    test_start = pd.Timestamp(year + 1, 1, 1)
    test_end = pd.Timestamp(year + args.test_years, 12, 1)
    current = monthly[monthly.month == dec].copy()
    universe = current.permno.to_numpy()
    train = monthly[(monthly.month >= train_start) & (monthly.month <= dec) & monthly.permno.isin(universe)]
    required = args.lookback_years * 12
    counts = train.groupby('permno').ret.count()
    valid = counts[counts >= required].index.to_numpy()
    current = current[current.permno.isin(valid)].sort_values('permno')
    assets = current.permno.to_numpy()
    n_assets = len(assets)
    train_months = pd.date_range(train_start, dec, freq='MS')
    test_months = pd.date_range(test_start, test_end, freq='MS')
    rtr = (
        train[train.permno.isin(assets)]
        .pivot(index='month', columns='permno', values='ret')
        .reindex(index=train_months, columns=assets)
        .to_numpy(float)
    )
    if np.isnan(rtr).any():
        raise RuntimeError('missing training returns')
    test = monthly[(monthly.month >= test_start) & (monthly.month <= test_end) & monthly.permno.isin(assets)]
    rte = (
        test.pivot(index='month', columns='permno', values='ret')
        .reindex(index=test_months, columns=assets)
        .fillna(0)
        .to_numpy(float)
    )
    mu, sigma = moments(rtr)
    mu_te, sigma_te = moments(rte)
    size = zscore(np.log(np.maximum(current.mktcap.to_numpy(float), 1)))
    mom = zscore(np.prod(1 + rtr[-12:], axis=0) - 1)
    vol = zscore(np.std(rtr[-24:], axis=0, ddof=1))
    components = min(5, rtr.shape[0], rtr.shape[1])
    pca_load = PCA(n_components=components, random_state=year).fit(rtr).components_.T
    sic_numeric = pd.to_numeric(current.siccd, errors='coerce').to_numpy(float)
    if np.isfinite(sic_numeric).mean() >= 0.8:
        sectors = np.array([ff12_industry(x) for x in sic_numeric], dtype=int)
        group_source = 'ff12'
    else:
        cluster_x = pca_load.copy()
        cluster_x = (cluster_x - cluster_x.mean(axis=0, keepdims=True)) / np.maximum(cluster_x.std(axis=0, keepdims=True), 1e-12)
        n_groups = min(12, n_assets)
        sectors = KMeans(n_clusters=n_groups, random_state=year, n_init=20).fit_predict(cluster_x)
        group_source = 'return_cluster'
    n_groups = int(sectors.max()) + 1
    onehot = np.eye(n_groups)[sectors]
    x = np.column_stack([1.3 * onehot, 0.6 * size, 0.6 * mom, -0.4 * vol, 0.5 * zscore(pca_load)])
    b = rng.normal(0, 0.55, size=(n_users, x.shape[1]))
    for u in range(n_users):
        fav = rng.choice(n_groups, size=2, replace=False)
        b[u, fav] += rng.uniform(1.5, 2.5, size=2)
        pool = [s for s in range(n_groups) if s not in fav]
        dislike = rng.choice(pool)
        b[u, dislike] -= rng.uniform(0.5, 1.2)
    popularity = 0.25 * zscore(np.log(np.maximum(current.mktcap.to_numpy(float), 1)))
    utilities = b @ x.T + popularity[None, :] + rng.normal(0, 0.25, size=(n_users, n_assets))
    holdings_n = min(args.holdings_n, n_assets - args.paper_top_k - 1)
    if holdings_n < 1 or n_assets - holdings_n < max(args.top_k, args.paper_top_k):
        raise RuntimeError('insufficient eligible assets')
    y = np.zeros((n_users, n_assets), dtype=float)
    relevant = []
    relevant_paper = []
    for u in range(n_users):
        held = rng.choice(n_assets, size=holdings_n, replace=False, p=softmax(utilities[u] / 0.85))
        y[u, held] = 1
        available = np.setdiff1d(np.arange(n_assets), held)
        ranked = available[np.argsort(-utilities[u, available])]
        relevant.append(ranked[:args.relevant_n])
        relevant_paper.append(ranked[:args.paper_top_k])
    w = y / y.sum(axis=1, keepdims=True)
    gamma = rng.choice(np.array([1.5, 3.0, 5.0]), size=n_users, p=[0.25, 0.5, 0.25])
    wm = implicit_als(y, args.factors, args.implicit_alpha, args.reg, args.iterations, seed + year)
    mv, y_tilde, c_tilde = mvecf_wmf(y, w, mu, sigma, args, seed + year)
    held_mask = y > 0
    wm[held_mask] = -np.inf
    mv[held_mask] = -np.inf
    available = np.isfinite(wm)
    return {
        'year': year,
        'seed': seed,
        'n_users': n_users,
        'n_assets': n_assets,
        'k': args.top_k,
        'delta': args.delta,
        'assets': assets,
        'tickers': current.ticker.fillna('').astype(str).to_numpy(),
        'mktcap': current.mktcap.to_numpy(float),
        'sectors': sectors,
        'group_source': group_source,
        'company_names': current.company_name.fillna(current.ticker).astype(str).to_numpy(),
        'isins': current['isin'].fillna('').astype(str).to_numpy(),
        'asset_keys': current.asset_key.fillna(current.permno.astype(str)).astype(str).to_numpy(),
        'test_years': args.test_years,
        'test_months': test_months,
        'W': w,
        'gamma': gamma,
        'utilities': utilities,
        'relevant': relevant,
        'relevant_paper': relevant_paper,
        'Y': y,
        'wm': wm,
        'zwm': row_zscore(wm, available),
        'mvecf': mv,
        'zmvecf': row_zscore(mv, available),
        'mvecf_y_tilde': y_tilde,
        'mvecf_c_tilde': c_tilde,
        'mu_tr': mu,
        'Sigma_tr': sigma,
        'mu_te': mu_te,
        'Sigma_te': sigma_te,
        'Rtr': rtr,
        'Rte': rte,
    }
