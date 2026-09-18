import argparse
import os
import random
from pathlib import Path

import numpy as np

from experiments.exp_trmv import ExpTRMV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, choices=['prepare', 'calibrate', 'test', 'report', 'all'])
    parser.add_argument('--model_id', default='TRMV-SF')
    parser.add_argument('--market', default='US_SP500', choices=['US_SP500', 'KR_KOSPI_200', 'JP_Nikkei_225', 'DE_DAX'])
    parser.add_argument('--archive_path', default='./wrds.zip')
    parser.add_argument('--root_path')
    parser.add_argument('--data_path', default='monthly.pkl')
    parser.add_argument('--result_path')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 4321, 7777])
    parser.add_argument('--calibration_years', nargs='+', type=int, default=[2004, 2005, 2006, 2007, 2008, 2009])
    parser.add_argument('--test_years_list', nargs='+', type=int, default=[2015, 2016, 2017, 2018, 2019, 2020])
    parser.add_argument('--users', type=int, default=300)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--paper_top_k', type=int)
    parser.add_argument('--holdings_n', type=int, default=10)
    parser.add_argument('--relevant_n', type=int, default=10)
    parser.add_argument('--lookback_years', type=int, default=5)
    parser.add_argument('--test_years', type=int, default=5)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--factors', type=int, default=20)
    parser.add_argument('--implicit_alpha', type=float, default=25.0)
    parser.add_argument('--positive_confidence', type=float, default=26.0)
    parser.add_argument('--negative_confidence', type=float, default=1.0)
    parser.add_argument('--reg', type=float, default=0.25)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--lambda_mv', type=float, default=10.0)
    parser.add_argument('--gamma_mv', type=float, default=3.0)
    parser.add_argument('--alpha_grid', nargs='+', type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--tau_grid', nargs='+', type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0])
    parser.add_argument('--mean_shrink', type=float, default=0.10)
    parser.add_argument('--trmv_plus_tau', type=float, default=1.0)
    parser.add_argument('--trmv_plus_cap', type=float, default=0.40)
    parser.add_argument('--buy_cap', type=float, default=0.50)
    parser.add_argument('--sell_cap', type=float, default=0.30)
    parser.add_argument('--cost_grid', nargs='+', type=float, default=[0.0, 10.0, 25.0, 50.0])
    parser.add_argument('--primary_cost_bps', type=float, default=25.0)
    parser.add_argument('--sims', type=int, default=100)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--bootstrap_seed', type=int, default=20260829)
    parser.add_argument('--representative_seed', type=int, default=1234)
    parser.add_argument('--representative_year', type=int, default=2020)
    parser.add_argument('--aum', type=float, default=100000.0)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.root_path is None:
        args.root_path = str(root / 'dataset' / args.market)
    if args.result_path is None:
        args.result_path = str(root / 'results' / args.market)
    if args.paper_top_k is None:
        args.paper_top_k = 10 if args.market == 'DE_DAX' else 20
    os.environ.setdefault('PYTHONHASHSEED', str(args.seed))
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    ExpTRMV(args).run()


if __name__ == '__main__':
    main()
