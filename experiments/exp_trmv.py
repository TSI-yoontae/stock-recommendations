from copy import copy
from pathlib import Path

import pandas as pd

from data_provider.context import build_context
from data_provider.prepare import prepare_market
from models.trmv_sf import equal_weights, marginal_utility_weights, self_financing_weights, topk, trmv_scores
from utils.portfolio import evaluate_calibration, evaluate_self_financing, evaluate_weighted, paper_protocol, prepare_baseline_cost_cache
from utils.report import choose_alpha, choose_temperature, combine_results, representative_outputs, summarize


class ExpTRMV:
    def __init__(self, args):
        self.args = args

    def prepare(self):
        prepare_market(self.args)

    def load_data(self):
        return pd.read_pickle(Path(self.args.root_path) / self.args.data_path)

    def calibrate(self):
        args = self.args
        monthly = self.load_data()
        cal_args = copy(args)
        cal_args.cost_grid = [args.primary_cost_bps]
        output = Path(args.result_path) / 'calibration'
        output.mkdir(parents=True, exist_ok=True)
        contexts = []
        for seed in args.seeds:
            for year in args.calibration_years:
                print('context', seed, year, flush=True)
                contexts.append(build_context(monthly, year, args.users, seed, args))
        alpha_rows = []
        for ctx in contexts:
            cache = prepare_baseline_cost_cache(ctx, cal_args.cost_grid)
            wm_rec = topk(ctx['wm'], ctx['k'])
            alpha_rows.append(evaluate_calibration(ctx, wm_rec, equal_weights(ctx['n_users'], ctx['k']), None, 'WMF', cache, cal_args))
            for alpha in args.alpha_grid:
                rec = topk(trmv_scores(ctx, alpha), ctx['k'])
                alpha_rows.append(evaluate_calibration(ctx, rec, equal_weights(ctx['n_users'], ctx['k']), None, 'TR-MV-a' + str(alpha), cache, cal_args))
        alpha_raw = pd.concat(alpha_rows, ignore_index=True)
        alpha_summary = summarize(alpha_raw)
        alpha = choose_alpha(alpha_summary)
        tau_rows = []
        for ctx in contexts:
            cache = prepare_baseline_cost_cache(ctx, cal_args.cost_grid)
            rec = topk(trmv_scores(ctx, alpha), ctx['k'])
            for temperature in args.tau_grid:
                q, sells, _, _ = self_financing_weights(ctx, rec, temperature, args)
                tau_rows.append(evaluate_calibration(ctx, rec, q, sells, 'TR-MV+SF-t' + str(temperature), cache, cal_args))
        tau_raw = pd.concat(tau_rows, ignore_index=True)
        tau_summary = summarize(tau_raw)
        temperature = choose_temperature(tau_summary)
        alpha_raw.to_pickle(output / 'alpha_user_raw.pkl')
        tau_raw.to_pickle(output / 'temperature_user_raw.pkl')
        alpha_summary.to_csv(output / 'alpha_summary.csv', index=False)
        tau_summary.to_csv(output / 'temperature_summary.csv', index=False)
        pd.DataFrame([{
            'selected_alpha': alpha,
            'selected_temperature': temperature,
            'lambda_mv': args.lambda_mv,
            'gamma_mv': args.gamma_mv,
            'mean_shrink': args.mean_shrink,
            'buy_cap': args.buy_cap,
            'sell_cap': args.sell_cap,
            'primary_cost_bps': args.primary_cost_bps,
        }]).to_csv(output / 'selected_rule.csv', index=False)
        print(alpha, temperature, flush=True)

    def test(self):
        args = self.args
        monthly = self.load_data()
        output = Path(args.result_path) / 'shards'
        output.mkdir(parents=True, exist_ok=True)
        rule = pd.read_csv(Path(args.result_path) / 'calibration' / 'selected_rule.csv').iloc[0]
        alpha = float(rule.selected_alpha)
        temperature = float(rule.selected_temperature)
        user_rows = []
        cost_rows = []
        paper_rows = []
        sample = None
        for year in args.test_years_list:
            print('test', args.seed, year, flush=True)
            ctx = build_context(monthly, year, args.users, args.seed, args)
            cache = prepare_baseline_cost_cache(ctx, args.cost_grid)
            wm_rec = topk(ctx['wm'], ctx['k'])
            mv_rec = topk(ctx['mvecf'], ctx['k'])
            trmv_rec = topk(trmv_scores(ctx, alpha), ctx['k'])
            equal = equal_weights(ctx['n_users'], ctx['k'])
            q_plus, _ = marginal_utility_weights(ctx, trmv_rec, args.trmv_plus_tau, args.mean_shrink, args.trmv_plus_cap)
            methods = [
                ('WMF', wm_rec, equal),
                ('MVECF', mv_rec, equal),
                ('TR-MV', trmv_rec, equal),
                ('TR-MV+', trmv_rec, q_plus),
            ]
            for method, recs, q in methods:
                result, costs = evaluate_weighted(ctx, recs, q, method, cache, args, args.sims)
                user_rows.append(result)
                cost_rows.append(costs)
            q_strong, sells_prop, _, _ = self_financing_weights(ctx, trmv_rec, temperature, args, True, False)
            result, costs = evaluate_self_financing(ctx, trmv_rec, q_strong, sells_prop, 'TR-MV+ strong-buy', cache, args, args.sims)
            user_rows.append(result)
            cost_rows.append(costs)
            _, sells_tilt, _, _ = self_financing_weights(ctx, trmv_rec, temperature, args, False, True)
            result, costs = evaluate_self_financing(ctx, trmv_rec, q_plus, sells_tilt, 'TR-MV+ funding-only', cache, args, args.sims)
            user_rows.append(result)
            cost_rows.append(costs)
            q_final, sells_final, buy_scores, sell_scores = self_financing_weights(ctx, trmv_rec, temperature, args, True, True)
            result, costs = evaluate_self_financing(ctx, trmv_rec, q_final, sells_final, 'TR-MV+SF', cache, args, args.sims)
            user_rows.append(result)
            cost_rows.append(costs)
            paper_rows.append(paper_protocol(ctx, ctx['wm'], 'WMF', args))
            paper_rows.append(paper_protocol(ctx, ctx['mvecf'], 'MVECF', args))
            if args.seed == args.representative_seed and year == args.representative_year:
                sample = (ctx, trmv_rec, q_final, sells_final, buy_scores, sell_scores, result)
        pd.concat(user_rows, ignore_index=True).to_pickle(output / ('test_user_seed' + str(args.seed) + '.pkl'))
        pd.concat(cost_rows, ignore_index=True).to_pickle(output / ('test_tc_seed' + str(args.seed) + '.pkl'))
        pd.concat(paper_rows, ignore_index=True).to_pickle(output / ('paper_seed' + str(args.seed) + '.pkl'))
        if sample is not None:
            orders, profiles = representative_outputs(sample, args)
            orders.to_csv(output / 'representative_orders.csv', index=False)
            profiles.to_csv(output / 'representative_profile_summary.csv', index=False)

    def report(self):
        combine_results(self.args)

    def all(self):
        self.prepare()
        self.calibrate()
        seed = self.args.seed
        for value in self.args.seeds:
            self.args.seed = value
            self.test()
        self.args.seed = seed
        self.report()

    def run(self):
        if self.args.task_name == 'prepare':
            self.prepare()
        elif self.args.task_name == 'calibrate':
            self.calibrate()
        elif self.args.task_name == 'test':
            self.test()
        elif self.args.task_name == 'report':
            self.report()
        elif self.args.task_name == 'all':
            self.all()
        else:
            raise ValueError(self.args.task_name)
