from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(df):
    columns = [c for c in df.columns if c not in ('year', 'seed', 'user', 'method')]
    return df.groupby('method')[columns].mean(numeric_only=True).reset_index()


def choose_alpha(summary):
    wm = summary[summary.method == 'WMF'].iloc[0]
    rows = summary[summary.method.str.startswith('TR-MV-a')].copy()
    rows['alpha'] = rows.method.str.extract(r'a([0-9.]+)$')[0].astype(float)
    feasible = rows[(rows.recall >= wm.recall) & (rows.ndcg >= wm.ndcg)]
    if feasible.empty:
        rows['shortfall'] = np.maximum(0, wm.recall - rows.recall) + np.maximum(0, wm.ndcg - rows.ndcg)
        return float(rows.sort_values(['shortfall', 'net_dce', 'alpha'], ascending=[True, False, True]).iloc[0].alpha)
    return float(feasible.sort_values(['net_dce', 'alpha'], ascending=[False, True]).iloc[0].alpha)


def choose_temperature(summary):
    rows = summary[summary.method.str.startswith('TR-MV+SF-t')].copy()
    rows['temperature'] = rows.method.str.extract(r't([0-9.]+)$')[0].astype(float)
    base = rows[rows.temperature == 0].iloc[0]
    feasible = rows[(rows.net_dsr >= base.net_dsr) & (rows.net_var_reduction >= base.net_var_reduction)]
    return float(feasible.sort_values(['net_dce', 'temperature'], ascending=[False, True]).iloc[0].temperature)


def bootstrap_years(user, args):
    metrics = [
        'recall',
        'ndcg',
        'net_dmean',
        'net_dce',
        'net_dsr',
        'net_var_reduction',
        'net_vol_reduction',
        'net_sortino_gain',
        'net_mdd_reduction',
        'net_cvar_reduction',
        'net_ce_safe',
        'net_sr_safe',
        'net_var_safe',
        'net_mdd_safe',
        'net_cvar_safe',
        'sim_net_dce',
        'sim_net_dsr',
        'sim_net_ce_safe',
        'sim_net_sr_safe',
        'annualized_tc_bps',
        'incremental_annualized_tc_bps',
    ]
    cell = user.groupby(['year', 'seed', 'method']).mean(numeric_only=True).reset_index()
    year_mean = cell.groupby(['year', 'method']).mean(numeric_only=True).reset_index()
    years = np.sort(year_mean.year.unique())
    proposed = year_mean[year_mean.method == 'TR-MV+SF'].set_index('year')
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    for baseline in ['WMF', 'MVECF', 'TR-MV', 'TR-MV+', 'TR-MV+ strong-buy', 'TR-MV+ funding-only']:
        base = year_mean[year_mean.method == baseline].set_index('year')
        if base.empty:
            continue
        for metric in metrics:
            if metric not in proposed.columns or metric not in base.columns:
                continue
            diff = (proposed.loc[years, metric] - base.loc[years, metric]).to_numpy(float)
            sample = diff[rng.integers(0, len(diff), size=(args.bootstrap, len(diff)))].mean(axis=1)
            rows.append({
                'proposed': 'TR-MV+SF',
                'baseline': baseline,
                'metric': metric,
                'mean_difference': float(diff.mean()),
                'ci_2_5': float(np.quantile(sample, 0.025)),
                'ci_97_5': float(np.quantile(sample, 0.975)),
                'positive_years': int(np.sum(diff > 0)),
                'n_years': len(diff),
            })
    return pd.DataFrame(rows)


def publication_table(summary):
    order = ['WMF', 'MVECF', 'TR-MV', 'TR-MV+', 'TR-MV+SF']
    rows = summary.set_index('method').loc[order].reset_index()
    return pd.DataFrame({
        'Method': rows.method,
        'Recall@5': rows.recall,
        'NDCG@5': rows.ndcg,
        'Max buy sleeve weight (%)': 100 * rows.max_weight,
        'Max trim sleeve weight (%)': 100 * rows.max_sell_weight,
        'Net Delta mean (x1e3)': 1000 * rows.net_dmean,
        'Net Delta CE (x1e3)': 1000 * rows.net_dce,
        'Net Delta Sharpe': rows.net_dsr,
        'Net variance reduction (x1e4)': 10000 * rows.net_var_reduction,
        'Net Sortino gain': rows.net_sortino_gain,
        'Net MDD reduction (x1e3)': 1000 * rows.net_mdd_reduction,
        'Net CVaR reduction (x1e3)': 1000 * rows.net_cvar_reduction,
        'CE improved users (%)': 100 * rows.net_ce_safe,
        'Sharpe improved users (%)': 100 * rows.net_sr_safe,
        'Variance improved users (%)': 100 * rows.net_var_safe,
        'MDD improved users (%)': 100 * rows.net_mdd_safe,
        'CVaR improved users (%)': 100 * rows.net_cvar_safe,
        'Annualized TC (bps)': rows.annualized_tc_bps,
        'Incremental annualized TC (bps)': rows.incremental_annualized_tc_bps,
        'Partial net Delta CE (x1e3)': 1000 * rows.sim_net_dce,
        'Partial net Delta Sharpe': rows.sim_net_dsr,
    })


def ablation_table(summary):
    order = ['TR-MV', 'TR-MV+', 'TR-MV+ strong-buy', 'TR-MV+ funding-only', 'TR-MV+SF']
    rows = summary.set_index('method').loc[order].reset_index()
    return pd.DataFrame({
        'Method': rows.method,
        'Recall@5': rows.recall,
        'NDCG@5': rows.ndcg,
        'Net Delta CE (x1e3)': 1000 * rows.net_dce,
        'Net Delta Sharpe': rows.net_dsr,
        'Net variance reduction (x1e4)': 10000 * rows.net_var_reduction,
        'Net MDD reduction (x1e3)': 1000 * rows.net_mdd_reduction,
        'Net CVaR reduction (x1e3)': 1000 * rows.net_cvar_reduction,
        'Annualized TC (bps)': rows.annualized_tc_bps,
    })


def paper_table(paper, args):
    rows = paper.groupby('method').mean(numeric_only=True).reset_index().set_index('method').loc[['WMF', 'MVECF']].reset_index()
    cost_key = 'net_' + str(float(args.primary_cost_bps)).rstrip('0').rstrip('.') + 'bps'
    return pd.DataFrame({
        'Method': rows.method,
        'MAP@' + str(args.paper_top_k): rows.map_at_k,
        'Recall@' + str(args.paper_top_k): rows.recall_at_k,
        'In-sample Delta mean': rows.in_sample_delta_mean,
        'In-sample Delta vol': rows.in_sample_delta_vol,
        'In-sample Delta Sharpe': rows.in_sample_delta_sr,
        'In-sample P(SR improves)': rows.in_sample_sr_improved,
        'Ex-post Delta mean': rows.ex_post_delta_mean,
        'Ex-post Delta vol': rows.ex_post_delta_vol,
        'Ex-post Delta Sharpe': rows.ex_post_delta_sr,
        'Ex-post P(SR improves)': rows.ex_post_sr_improved,
        'Net Delta mean': rows[cost_key + '_delta_mean'],
        'Net Delta vol': rows[cost_key + '_delta_vol'],
        'Net Delta Sharpe': rows[cost_key + '_delta_sr'],
        'Net P(SR improves)': rows[cost_key + '_sr_improved'],
        'Annualized TC (bps)': rows[cost_key + '_annualized_tc_bps'],
    })


def representative_outputs(sample, args):
    ctx, recs, qmat, sells, buy_scores, sell_scores, metrics = sample
    rows = []
    profiles = []
    for gamma_value, label in [(1.5, 'low'), (3.0, 'medium'), (5.0, 'high')]:
        candidates = np.flatnonzero(np.isclose(ctx['gamma'], gamma_value))
        values = metrics.set_index('user').loc[candidates, 'net_dce']
        median = float(values.median())
        u = int((values - median).abs().idxmin())
        rec = np.asarray(recs[u], dtype=int)
        q = qmat[u]
        held, sell_weight = sells[u]
        held = np.asarray(held, dtype=int)
        metric = metrics[metrics.user == u].iloc[0]
        for action, indices, weights, scores in [
            ('BUY', rec, q, buy_scores[u]),
            ('TRIM', held, sell_weight, sell_scores[u]),
        ]:
            order = np.argsort(-weights)
            for rank, j in enumerate(order, 1):
                asset = int(indices[j])
                weight = float(weights[j])
                portfolio_change = ctx['delta'] * weight
                amount = args.aum * portfolio_change
                rows.append({
                    'market': args.market,
                    'profile': label,
                    'gamma': gamma_value,
                    'recommendation_year': args.representative_year,
                    'representative_user_id': u,
                    'action': action,
                    'action_rank': rank,
                    'ticker': str(ctx['tickers'][asset]),
                    'company_name': str(ctx['company_names'][asset]),
                    'asset_id': str(ctx['asset_keys'][asset]),
                    'isin': str(ctx['isins'][asset]),
                    'action_sleeve_weight_pct': 100 * weight,
                    'total_portfolio_change_pct': (1 if action == 'BUY' else -1) * 100 * portfolio_change,
                    'order_amount': amount,
                    'estimated_one_way_cost': amount * args.primary_cost_bps / 10000,
                    'marginal_score': float(scores[j]),
                })
        profiles.append({
            'market': args.market,
            'profile': label,
            'gamma': gamma_value,
            'representative_user_id': u,
            'buy_sleeve_pct': 100 * ctx['delta'],
            'trim_sleeve_pct': 100 * ctx['delta'],
            'max_single_buy_portfolio_pct': 100 * ctx['delta'] * q.max(),
            'max_single_trim_portfolio_pct': 100 * ctx['delta'] * sell_weight.max(),
            'initial_total_turnover_pct': 200 * ctx['delta'],
            'initial_total_cost': args.aum * 2 * ctx['delta'] * args.primary_cost_bps / 10000,
            'net_delta_ce': float(metric.net_dce),
            'net_delta_sharpe': float(metric.net_dsr),
            'net_variance_reduction': float(metric.net_var_reduction),
            'annualized_tc_bps': float(metric.annualized_tc_bps),
        })
    return pd.DataFrame(rows), pd.DataFrame(profiles)

def make_figures(summary, tc_summary, out):
    figure_path = Path(out) / 'figures'
    figure_path.mkdir(parents=True, exist_ok=True)
    main = summary[summary.method.isin(['WMF', 'MVECF', 'TR-MV', 'TR-MV+', 'TR-MV+SF'])].copy()
    plt.figure(figsize=(7, 5))
    for _, row in main.iterrows():
        plt.scatter(row.ndcg, 1000 * row.net_dce, s=70)
        plt.text(row.ndcg, 1000 * row.net_dce, row.method)
    plt.xlabel('NDCG@5')
    plt.ylabel('Net Delta CE x 1e3')
    plt.tight_layout()
    plt.savefig(figure_path / 'preference_net_ce_frontier.png', dpi=200)
    plt.close()
    ablation = summary[summary.method.isin(['TR-MV', 'TR-MV+', 'TR-MV+ strong-buy', 'TR-MV+ funding-only', 'TR-MV+SF'])].copy()
    plt.figure(figsize=(8, 5))
    plt.bar(ablation.method, 1000 * ablation.net_dce)
    plt.ylabel('Net Delta CE x 1e3')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(figure_path / 'ablation_net_ce.png', dpi=200)
    plt.close()
    selected = tc_summary[tc_summary.method.isin(['WMF', 'MVECF', 'TR-MV+', 'TR-MV+SF'])]
    plt.figure(figsize=(7, 5))
    for method, group in selected.groupby('method'):
        group = group.sort_values('cost_bps')
        plt.plot(group.cost_bps, 1000 * group.net_dce, marker='o', label=method)
    plt.xlabel('One-way transaction cost bps')
    plt.ylabel('Net Delta CE x 1e3')
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path / 'cost_sensitivity_net_ce.png', dpi=200)
    plt.close()


def combine_results(args):
    shard_path = Path(args.result_path) / 'shards'
    output = Path(args.result_path) / 'final'
    output.mkdir(parents=True, exist_ok=True)
    user = pd.concat([pd.read_pickle(p) for p in sorted(shard_path.glob('test_user_seed*.pkl'))], ignore_index=True)
    tc = pd.concat([pd.read_pickle(p) for p in sorted(shard_path.glob('test_tc_seed*.pkl'))], ignore_index=True)
    paper = pd.concat([pd.read_pickle(p) for p in sorted(shard_path.glob('paper_seed*.pkl'))], ignore_index=True)
    user.to_pickle(output / 'test_user_raw.pkl')
    tc.to_pickle(output / 'test_tc_raw.pkl')
    paper.to_pickle(output / 'paper_raw.pkl')
    summary = summarize(user)
    tc_summary = tc.groupby(['method', 'cost_bps']).mean(numeric_only=True).reset_index()
    summary.to_csv(output / 'test_summary.csv', index=False)
    tc_summary.to_csv(output / 'transaction_cost_sensitivity.csv', index=False)
    publication_table(summary).to_csv(output / 'publication_main_table_25bps.csv', index=False)
    ablation_table(summary).to_csv(output / 'ablation_table.csv', index=False)
    paper_table(paper, args).to_csv(output / 'paper_faithful_mvecf_table.csv', index=False)
    bootstrap_years(user, args).to_csv(output / 'paired_year_bootstrap.csv', index=False)
    selected = tc_summary[tc_summary.method.isin(['WMF', 'MVECF', 'TR-MV', 'TR-MV+', 'TR-MV+SF'])].copy()
    selected['Net Delta CE (x1e3)'] = 1000 * selected.net_dce
    selected['Net Delta Sharpe'] = selected.net_dsr
    selected['Net variance reduction (x1e4)'] = 10000 * selected.net_var_reduction
    selected[['method', 'cost_bps', 'Net Delta CE (x1e3)', 'Net Delta Sharpe', 'Net variance reduction (x1e4)', 'annualized_tc_bps']].to_csv(output / 'cost_sensitivity_table.csv', index=False)
    rule = Path(args.result_path) / 'calibration' / 'selected_rule.csv'
    pd.read_csv(rule).to_csv(output / 'selected_rule.csv', index=False)
    for name in ['representative_orders.csv', 'representative_profile_summary.csv']:
        source = shard_path / name
        if source.exists():
            pd.read_csv(source).to_csv(output / name, index=False)
    year_summary = user.groupby(['year', 'method']).mean(numeric_only=True).reset_index()
    gamma_summary = user.groupby(['gamma', 'method']).mean(numeric_only=True).reset_index()
    year_summary.to_csv(output / 'year_breakdown.csv', index=False)
    gamma_summary.to_csv(output / 'gamma_breakdown.csv', index=False)
    make_figures(summary, tc_summary, output)
    print(publication_table(summary).to_string(index=False))
