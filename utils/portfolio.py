import math

import numpy as np
import pandas as pd
from scipy.special import expit

from models.trmv_sf import target_self_financing, target_weighted, topk
from utils.metrics import average_precision_at_k, cvar_loss, max_drawdown, path_statistics, sortino_ratio


def gross_path_and_turnover(returns, target, initial):
    support = np.flatnonzero((target > 1e-15) | (initial > 1e-15))
    target_support = target[support]
    initial_support = initial[support]
    asset_returns = returns[:, support]
    gross = asset_returns @ target_support
    denom = np.maximum(1 + gross, 1e-12)
    drift = target_support[None, :] * (1 + asset_returns) / denom[:, None]
    rebalance_l1 = np.abs(drift - target_support[None, :]).sum(axis=1)
    initial_l1 = float(np.abs(target_support - initial_support).sum())
    return gross, rebalance_l1, initial_l1


def apply_transaction_costs(gross, rebalance_l1, initial_l1, cost_bps):
    cost = cost_bps / 10000
    growth = (1 + np.asarray(gross, dtype=float)) * (1 - cost * np.asarray(rebalance_l1, dtype=float))
    growth = np.maximum(growth, 1e-12)
    growth[0] *= max(1 - cost * initial_l1, 1e-12)
    return growth - 1


def prepare_baseline_cost_cache(ctx, cost_grid):
    cache = []
    for u in range(ctx['n_users']):
        gross, rebalance_l1, initial_l1 = gross_path_and_turnover(ctx['Rte'], ctx['W'][u], ctx['W'][u])
        gross_stats = path_statistics(gross, ctx['gamma'][u])
        per_cost = {}
        for bps in cost_grid:
            net = apply_transaction_costs(gross, rebalance_l1, initial_l1, bps)
            stats = path_statistics(net, ctx['gamma'][u])
            drag = 1 - stats['terminal_wealth'] / max(gross_stats['terminal_wealth'], 1e-12)
            per_cost[float(bps)] = {
                'stats': stats,
                'cost_drag': float(drag),
                'annualized_tc_bps': float(bps) * 12 * float(np.mean(rebalance_l1)),
            }
        cache.append({
            'gross': gross,
            'rebalance_l1': rebalance_l1,
            'initial_l1': initial_l1,
            'gross_stats': gross_stats,
            'per_cost': per_cost,
        })
    return cache


def add_cost_rows(row, tc_rows, ctx, u, method, gross_path, rebalance_l1, initial_l1, gross_stats, base_cache, cost_grid, primary_cost):
    for bps0 in cost_grid:
        bps = float(bps0)
        net_path = apply_transaction_costs(gross_path, rebalance_l1, initial_l1, bps)
        target_stats = path_statistics(net_path, ctx['gamma'][u])
        base_stats = base_cache[u]['per_cost'][bps]['stats']
        target_drag = 1 - target_stats['terminal_wealth'] / max(gross_stats['terminal_wealth'], 1e-12)
        base_drag = base_cache[u]['per_cost'][bps]['cost_drag']
        annual_tc = bps * (initial_l1 / ctx['test_years'] + 12 * float(np.mean(rebalance_l1)))
        base_annual_tc = base_cache[u]['per_cost'][bps]['annualized_tc_bps']
        tc_row = {
            'year': ctx['year'],
            'seed': ctx['seed'],
            'user': u,
            'method': method,
            'cost_bps': bps,
            'initial_cost_bps_portfolio': bps * initial_l1,
            'annualized_tc_bps': annual_tc,
            'incremental_annualized_tc_bps': annual_tc - base_annual_tc,
            'five_year_cost_drag_bps': 10000 * target_drag,
            'incremental_five_year_cost_drag_bps': 10000 * (target_drag - base_drag),
            'net_dmean': target_stats['mean'] - base_stats['mean'],
            'net_dce': target_stats['ce'] - base_stats['ce'],
            'net_dsr': target_stats['sr'] - base_stats['sr'],
            'net_var_reduction': base_stats['var'] - target_stats['var'],
            'net_vol_reduction': base_stats['vol'] - target_stats['vol'],
            'net_sortino_gain': target_stats['sortino'] - base_stats['sortino'],
            'net_mdd_reduction': base_stats['mdd'] - target_stats['mdd'],
            'net_cvar_reduction': base_stats['cvar'] - target_stats['cvar'],
            'net_ce_safe': float(target_stats['ce'] >= base_stats['ce']),
            'net_sr_safe': float(target_stats['sr'] >= base_stats['sr']),
            'net_var_safe': float(target_stats['var'] <= base_stats['var']),
            'net_sortino_safe': float(target_stats['sortino'] >= base_stats['sortino']),
            'net_mdd_safe': float(target_stats['mdd'] <= base_stats['mdd']),
            'net_cvar_safe': float(target_stats['cvar'] <= base_stats['cvar']),
        }
        tc_rows.append(tc_row)
        if abs(bps - primary_cost) < 1e-12:
            row.update(tc_row)
            row['primary_cost_bps'] = bps


def evaluate_weighted(ctx, rec_lists, qmat, method, base_cache, args, sims=0):
    w = ctx['W']
    gamma = ctx['gamma']
    mu = ctx['mu_te']
    sigma = ctx['Sigma_te']
    returns = ctx['Rte']
    utilities = ctx['utilities']
    relevant = ctx['relevant']
    delta = ctx['delta']
    k = ctx['k']
    n_users = ctx['n_users']
    ws = w @ sigma
    mean0 = w @ mu
    var0 = np.einsum('ij,ij->i', ws, w)
    ce0 = mean0 - 0.5 * gamma * var0
    sr0 = mean0 / np.sqrt(np.maximum(var0, 1e-12))
    diag = np.diag(sigma)
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = disc.sum()
    if sims > 0:
        rng = np.random.default_rng(10000000 + ctx['seed'] * 100 + ctx['year'])
        accept_draw = rng.random((n_users, sims, k))
        delta_draw = rng.random((n_users, sims))
        weight_noise = rng.exponential(1, size=(n_users, sims, k))
    rows = []
    tc_rows = []
    for u, rec0 in enumerate(rec_lists):
        rec = np.asarray(rec0, dtype=int)
        q = np.asarray(qmat[u], dtype=float)
        rel = np.isin(rec, relevant[u]).astype(float)
        hits = rel.sum()
        cov_wr = ws[u, rec]
        var_single = (1 - delta) ** 2 * var0[u] + 2 * delta * (1 - delta) * cov_wr + delta ** 2 * diag[rec]
        mean_single = (1 - delta) * mean0[u] + delta * mu[rec]
        ce_single = mean_single - 0.5 * gamma[u] * var_single
        sr_single = mean_single / np.sqrt(np.maximum(var_single, 1e-12))
        dce_single = ce_single - ce0[u]
        dsr_single = sr_single - sr0[u]
        dvar_single = var_single - var0[u]
        mean_q = q @ mu[rec]
        cov_wq = q @ cov_wr
        var_q = q @ sigma[np.ix_(rec, rec)] @ q
        mean_full = (1 - delta) * mean0[u] + delta * mean_q
        var_full = (1 - delta) ** 2 * var0[u] + 2 * delta * (1 - delta) * cov_wq + delta ** 2 * var_q
        ce_full = mean_full - 0.5 * gamma[u] * var_full
        sr_full = mean_full / math.sqrt(max(var_full, 1e-12))
        initial_path = returns @ w[u]
        final_path = (1 - delta) * initial_path + delta * (returns[:, rec] @ q)
        vol0 = float(np.std(initial_path, ddof=1) * math.sqrt(12))
        vol1 = float(np.std(final_path, ddof=1) * math.sqrt(12))
        sortino0 = sortino_ratio(initial_path)
        sortino1 = sortino_ratio(final_path)
        mdd0 = max_drawdown(initial_path)
        mdd1 = max_drawdown(final_path)
        cvar0 = cvar_loss(initial_path)
        cvar1 = cvar_loss(final_path)
        target = target_weighted(ctx, u, rec, q)
        gross_path, rebalance_l1, initial_l1 = gross_path_and_turnover(returns, target, w[u])
        gross_stats = path_statistics(gross_path, gamma[u])
        row = {
            'year': ctx['year'],
            'seed': ctx['seed'],
            'user': u,
            'method': method,
            'gamma': float(gamma[u]),
            'recall': float(hits / len(relevant[u])),
            'precision': float(hits / len(rec)),
            'ndcg': float((rel * disc).sum() / idcg),
            'mean_pref': float(utilities[u, rec].mean()),
            'max_weight': float(q.max()),
            'min_weight': float(q.min()),
            'herfindahl': float(q @ q),
            'max_sell_weight': np.nan,
            'min_sell_weight': np.nan,
            'sell_herfindahl': np.nan,
            'max_total_portfolio_buy': float(delta * q.max()),
            'max_total_portfolio_trim': np.nan,
            'singleton_ce_safe': float(np.min(dce_single) >= 0),
            'singleton_sr_safe': float(np.min(dsr_single) >= 0),
            'singleton_var_safe': float(np.max(dvar_single) <= 0),
            'worst_singleton_dce': float(np.min(dce_single)),
            'worst_singleton_dsr': float(np.min(dsr_single)),
            'worst_singleton_dvar': float(np.max(dvar_single)),
            'full_dmean': float(mean_full - mean0[u]),
            'full_dce': float(ce_full - ce0[u]),
            'full_dsr': float(sr_full - sr0[u]),
            'full_var_reduction': float(var0[u] - var_full),
            'full_vol_reduction': float(vol0 - vol1),
            'full_sortino_gain': float(sortino1 - sortino0),
            'full_mdd_reduction': float(mdd0 - mdd1),
            'full_cvar_reduction': float(cvar0 - cvar1),
            'full_ce_safe': float(ce_full >= ce0[u]),
            'full_sr_safe': float(sr_full >= sr0[u]),
            'full_var_safe': float(var_full <= var0[u]),
            'full_sortino_safe': float(sortino1 >= sortino0),
            'full_mdd_safe': float(mdd1 <= mdd0),
            'full_cvar_safe': float(cvar1 <= cvar0),
            'initial_l1_traded': float(initial_l1),
            'annual_rebalance_l1': float(12 * np.mean(rebalance_l1)),
        }
        add_cost_rows(row, tc_rows, ctx, u, method, gross_path, rebalance_l1, initial_l1, gross_stats, base_cache, args.cost_grid, args.primary_cost_bps)
        if sims > 0:
            accept_prob = expit((utilities[u, rec] - np.median(utilities[u, rec])) / 0.8)
            accepted = accept_draw[u] < accept_prob[None, :]
            empty = ~accepted.any(axis=1)
            accepted[empty, np.argmax(utilities[u, rec])] = True
            weights = q[None, :] * weight_noise[u] * accepted
            weights /= weights.sum(axis=1, keepdims=True)
            dd = delta * delta_draw[u]
            mean_qs = weights @ mu[rec]
            cov_wqs = weights @ cov_wr
            s_rr = sigma[np.ix_(rec, rec)]
            var_qs = np.einsum('bi,ij,bj->b', weights, s_rr, weights)
            means = (1 - dd) * mean0[u] + dd * mean_qs
            vars_ = (1 - dd) ** 2 * var0[u] + 2 * dd * (1 - dd) * cov_wqs + dd ** 2 * var_qs
            ces = means - 0.5 * gamma[u] * vars_
            srs = means / np.sqrt(np.maximum(vars_, 1e-12))
            dce = ces - ce0[u]
            dsr = srs - sr0[u]
            var_reduction = var0[u] - vars_
            cost = args.primary_cost_bps / 10000
            implementation_cost = np.minimum(2 * cost * dd, 0.999999)
            annual_cost_drag = 1 - (1 - implementation_cost) ** (1 / ctx['test_years'])
            net_means = means - annual_cost_drag
            net_ces = net_means - 0.5 * gamma[u] * vars_
            net_srs = net_means / np.sqrt(np.maximum(vars_, 1e-12))
            row.update({
                'sim_dmean': float(np.mean(means - mean0[u])),
                'sim_dce': float(dce.mean()),
                'sim_dsr': float(dsr.mean()),
                'sim_var_reduction': float(var_reduction.mean()),
                'sim_ce_safe': float(np.mean(dce >= 0)),
                'sim_sr_safe': float(np.mean(dsr >= 0)),
                'sim_var_safe': float(np.mean(var_reduction >= 0)),
                'sim_net_dmean': float(np.mean(net_means - mean0[u])),
                'sim_net_dce': float((net_ces - ce0[u]).mean()),
                'sim_net_dsr': float((net_srs - sr0[u]).mean()),
                'sim_net_ce_safe': float(np.mean(net_ces >= ce0[u])),
                'sim_net_sr_safe': float(np.mean(net_srs >= sr0[u])),
                'sim_mean_initial_cost_bps': float(np.mean(implementation_cost) * 10000),
            })
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(tc_rows)


def evaluate_self_financing(ctx, rec_lists, qmat, sells, method, base_cache, args, sims=0):
    w = ctx['W']
    gamma = ctx['gamma']
    mu = ctx['mu_te']
    sigma = ctx['Sigma_te']
    returns = ctx['Rte']
    utilities = ctx['utilities']
    relevant = ctx['relevant']
    delta = ctx['delta']
    k = ctx['k']
    n_users = ctx['n_users']
    ws = w @ sigma
    mean0 = w @ mu
    var0 = np.einsum('ij,ij->i', ws, w)
    ce0 = mean0 - 0.5 * gamma * var0
    sr0 = mean0 / np.sqrt(np.maximum(var0, 1e-12))
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = disc.sum()
    if sims > 0:
        rng = np.random.default_rng(10000000 + ctx['seed'] * 100 + ctx['year'])
        accept_draw = rng.random((n_users, sims, k))
        delta_draw = rng.random((n_users, sims))
        weight_noise = rng.exponential(1, size=(n_users, sims, k))
    rows = []
    tc_rows = []
    for u, rec0 in enumerate(rec_lists):
        rec = np.asarray(rec0, dtype=int)
        q = np.asarray(qmat[u], dtype=float)
        held, s = sells[u]
        held = np.asarray(held, dtype=int)
        s = np.asarray(s, dtype=float)
        rel = np.isin(rec, relevant[u]).astype(float)
        hits = rel.sum()
        target = target_self_financing(ctx, u, rec, q, (held, s))
        mean_full = float(mu @ target)
        var_full = float(target @ sigma @ target)
        ce_full = mean_full - 0.5 * gamma[u] * var_full
        sr_full = mean_full / math.sqrt(max(var_full, 1e-12))
        mean_single = []
        var_single = []
        for item in rec:
            single = w[u].copy()
            single[held] -= delta * s
            single[item] += delta
            mean_single.append(float(mu @ single))
            var_single.append(float(single @ sigma @ single))
        mean_single = np.asarray(mean_single)
        var_single = np.asarray(var_single)
        ce_single = mean_single - 0.5 * gamma[u] * var_single
        sr_single = mean_single / np.sqrt(np.maximum(var_single, 1e-12))
        dce_single = ce_single - ce0[u]
        dsr_single = sr_single - sr0[u]
        dvar_single = var_single - var0[u]
        initial_path = returns @ w[u]
        final_path = returns @ target
        vol0 = float(np.std(initial_path, ddof=1) * math.sqrt(12))
        vol1 = float(np.std(final_path, ddof=1) * math.sqrt(12))
        sortino0 = sortino_ratio(initial_path)
        sortino1 = sortino_ratio(final_path)
        mdd0 = max_drawdown(initial_path)
        mdd1 = max_drawdown(final_path)
        cvar0 = cvar_loss(initial_path)
        cvar1 = cvar_loss(final_path)
        gross_path, rebalance_l1, initial_l1 = gross_path_and_turnover(returns, target, w[u])
        gross_stats = path_statistics(gross_path, gamma[u])
        row = {
            'year': ctx['year'],
            'seed': ctx['seed'],
            'user': u,
            'method': method,
            'gamma': float(gamma[u]),
            'recall': float(hits / len(relevant[u])),
            'precision': float(hits / len(rec)),
            'ndcg': float((rel * disc).sum() / idcg),
            'mean_pref': float(utilities[u, rec].mean()),
            'max_weight': float(q.max()),
            'min_weight': float(q.min()),
            'herfindahl': float(q @ q),
            'max_sell_weight': float(s.max()),
            'min_sell_weight': float(s.min()),
            'sell_herfindahl': float(s @ s),
            'max_total_portfolio_buy': float(delta * q.max()),
            'max_total_portfolio_trim': float(delta * s.max()),
            'singleton_ce_safe': float(np.min(dce_single) >= 0),
            'singleton_sr_safe': float(np.min(dsr_single) >= 0),
            'singleton_var_safe': float(np.max(dvar_single) <= 0),
            'worst_singleton_dce': float(np.min(dce_single)),
            'worst_singleton_dsr': float(np.min(dsr_single)),
            'worst_singleton_dvar': float(np.max(dvar_single)),
            'full_dmean': float(mean_full - mean0[u]),
            'full_dce': float(ce_full - ce0[u]),
            'full_dsr': float(sr_full - sr0[u]),
            'full_var_reduction': float(var0[u] - var_full),
            'full_vol_reduction': float(vol0 - vol1),
            'full_sortino_gain': float(sortino1 - sortino0),
            'full_mdd_reduction': float(mdd0 - mdd1),
            'full_cvar_reduction': float(cvar0 - cvar1),
            'full_ce_safe': float(ce_full >= ce0[u]),
            'full_sr_safe': float(sr_full >= sr0[u]),
            'full_var_safe': float(var_full <= var0[u]),
            'full_sortino_safe': float(sortino1 >= sortino0),
            'full_mdd_safe': float(mdd1 <= mdd0),
            'full_cvar_safe': float(cvar1 <= cvar0),
            'initial_l1_traded': float(initial_l1),
            'annual_rebalance_l1': float(12 * np.mean(rebalance_l1)),
        }
        add_cost_rows(row, tc_rows, ctx, u, method, gross_path, rebalance_l1, initial_l1, gross_stats, base_cache, args.cost_grid, args.primary_cost_bps)
        if sims > 0:
            accept_prob = expit((utilities[u, rec] - np.median(utilities[u, rec])) / 0.8)
            accepted = accept_draw[u] < accept_prob[None, :]
            empty = ~accepted.any(axis=1)
            accepted[empty, np.argmax(utilities[u, rec])] = True
            buy_weights = q[None, :] * weight_noise[u] * accepted
            buy_weights /= buy_weights.sum(axis=1, keepdims=True)
            dd = delta * delta_draw[u]
            mean_buy = buy_weights @ mu[rec]
            mean_sell = float(s @ mu[held])
            cov_w_buy = buy_weights @ ws[u, rec]
            cov_w_sell = float(s @ ws[u, held])
            s_rr = sigma[np.ix_(rec, rec)]
            s_hh = sigma[np.ix_(held, held)]
            s_rh = sigma[np.ix_(rec, held)]
            var_buy = np.einsum('bi,ij,bj->b', buy_weights, s_rr, buy_weights)
            var_sell = float(s @ s_hh @ s)
            cross = (buy_weights @ s_rh) @ s
            cov_direction = cov_w_buy - cov_w_sell
            var_direction = var_buy + var_sell - 2 * cross
            means = mean0[u] + dd * (mean_buy - mean_sell)
            vars_ = var0[u] + 2 * dd * cov_direction + dd ** 2 * var_direction
            ces = means - 0.5 * gamma[u] * vars_
            srs = means / np.sqrt(np.maximum(vars_, 1e-12))
            var_reduction = var0[u] - vars_
            cost = args.primary_cost_bps / 10000
            implementation_cost = np.minimum(2 * cost * dd, 0.999999)
            annual_cost_drag = 1 - (1 - implementation_cost) ** (1 / ctx['test_years'])
            net_means = means - annual_cost_drag
            net_ces = net_means - 0.5 * gamma[u] * vars_
            net_srs = net_means / np.sqrt(np.maximum(vars_, 1e-12))
            row.update({
                'sim_dmean': float(np.mean(means - mean0[u])),
                'sim_dce': float((ces - ce0[u]).mean()),
                'sim_dsr': float((srs - sr0[u]).mean()),
                'sim_var_reduction': float(var_reduction.mean()),
                'sim_ce_safe': float(np.mean(ces >= ce0[u])),
                'sim_sr_safe': float(np.mean(srs >= sr0[u])),
                'sim_var_safe': float(np.mean(var_reduction >= 0)),
                'sim_net_dmean': float(np.mean(net_means - mean0[u])),
                'sim_net_dce': float((net_ces - ce0[u]).mean()),
                'sim_net_dsr': float((net_srs - sr0[u]).mean()),
                'sim_net_ce_safe': float(np.mean(net_ces >= ce0[u])),
                'sim_net_sr_safe': float(np.mean(net_srs >= sr0[u])),
                'sim_mean_initial_cost_bps': float(np.mean(implementation_cost) * 10000),
            })
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(tc_rows)


def paper_protocol(ctx, scores, method, args):
    rows = []
    recs = topk(scores, args.paper_top_k)
    for u, rec0 in enumerate(recs):
        rec = np.asarray(rec0, dtype=int)
        held = np.flatnonzero(ctx['Y'][u] > 0)
        initial = np.zeros(ctx['n_assets'], dtype=float)
        initial[held] = 1 / len(held)
        final = np.zeros(ctx['n_assets'], dtype=float)
        union = np.r_[held, rec]
        final[union] = 1 / len(union)
        relevant = np.asarray(ctx['relevant_paper'][u])
        hit = np.isin(rec, relevant).astype(float)
        record = {
            'year': ctx['year'],
            'seed': ctx['seed'],
            'user': u,
            'method': method,
            'paper_k': args.paper_top_k,
            'map_at_k': average_precision_at_k(rec, relevant, args.paper_top_k),
            'recall_at_k': float(hit.sum() / len(relevant)),
        }
        for label, returns in [('in_sample', ctx['Rtr']), ('ex_post', ctx['Rte'])]:
            stats0 = path_statistics(returns @ initial, args.gamma_mv)
            stats1 = path_statistics(returns @ final, args.gamma_mv)
            record[label + '_delta_mean'] = stats1['mean'] - stats0['mean']
            record[label + '_delta_vol'] = stats1['vol'] - stats0['vol']
            record[label + '_delta_sr'] = stats1['sr'] - stats0['sr']
            record[label + '_sr_improved'] = float(stats1['sr'] > stats0['sr'])
        gross0, rebalance0, initial0 = gross_path_and_turnover(ctx['Rte'], initial, initial)
        gross1, rebalance1, initial1 = gross_path_and_turnover(ctx['Rte'], final, initial)
        for bps in args.cost_grid:
            stats0 = path_statistics(apply_transaction_costs(gross0, rebalance0, initial0, bps), args.gamma_mv)
            stats1 = path_statistics(apply_transaction_costs(gross1, rebalance1, initial1, bps), args.gamma_mv)
            key = 'net_' + str(float(bps)).rstrip('0').rstrip('.') + 'bps'
            record[key + '_delta_mean'] = stats1['mean'] - stats0['mean']
            record[key + '_delta_vol'] = stats1['vol'] - stats0['vol']
            record[key + '_delta_sr'] = stats1['sr'] - stats0['sr']
            record[key + '_sr_improved'] = float(stats1['sr'] > stats0['sr'])
            record[key + '_annualized_tc_bps'] = bps * (initial1 / ctx['test_years'] + 12 * float(np.mean(rebalance1)))
        rows.append(record)
    return pd.DataFrame(rows)


def evaluate_calibration(ctx, rec_lists, qmat, sells, method, base_cache, args):
    k = ctx['k']
    disc = 1 / np.log2(np.arange(2, k + 2))
    idcg = disc.sum()
    rows = []
    for u, rec0 in enumerate(rec_lists):
        rec = np.asarray(rec0, dtype=int)
        q = np.asarray(qmat[u], dtype=float)
        rel = np.isin(rec, ctx['relevant'][u]).astype(float)
        if sells is None:
            target = target_weighted(ctx, u, rec, q)
        else:
            target = target_self_financing(ctx, u, rec, q, sells[u])
        gross, rebalance_l1, initial_l1 = gross_path_and_turnover(ctx['Rte'], target, ctx['W'][u])
        net = apply_transaction_costs(gross, rebalance_l1, initial_l1, args.primary_cost_bps)
        target_stats = path_statistics(net, ctx['gamma'][u])
        base_stats = base_cache[u]['per_cost'][float(args.primary_cost_bps)]['stats']
        rows.append({
            'year': ctx['year'],
            'seed': ctx['seed'],
            'user': u,
            'method': method,
            'recall': float(rel.sum() / len(ctx['relevant'][u])),
            'ndcg': float((rel * disc).sum() / idcg),
            'net_dce': target_stats['ce'] - base_stats['ce'],
            'net_dsr': target_stats['sr'] - base_stats['sr'],
            'net_var_reduction': base_stats['var'] - target_stats['var'],
        })
    return pd.DataFrame(rows)
