# TRMV-SF

Compact code for WMF, paper-faithful MVECF, TR-MV, TR-MV+, and self-financing TR-MV+ on the four markets in `wrds.zip`.

## Structure

```text
TRMV-SF/
├── run.py
├── data_provider/
│   ├── prepare.py
│   └── context.py
├── models/
│   ├── wmf.py
│   ├── mvecf.py
│   └── trmv_sf.py
├── experiments/
│   └── exp_trmv.py
├── utils/
│   ├── parquet.py
│   ├── metrics.py
│   ├── portfolio.py
│   └── report.py
├── scripts/
│   ├── US_SP500.sh
│   ├── KR_KOSPI_200.sh
│   ├── JP_Nikkei_225.sh
│   ├── DE_DAX.sh
│   └── ALL_MARKETS.sh
├── dataset/
├── results/
└── requirements.txt
```

## Environment

```bash
pip install -r requirements.txt
```

## Full experiment

```bash
bash scripts/US_SP500.sh /path/to/wrds.zip
bash scripts/KR_KOSPI_200.sh /path/to/wrds.zip
bash scripts/JP_Nikkei_225.sh /path/to/wrds.zip
bash scripts/DE_DAX.sh /path/to/wrds.zip
```

```bash
bash scripts/ALL_MARKETS.sh /path/to/wrds.zip
```

## Stage execution

```bash
python -u run.py --task_name prepare --market US_SP500 --archive_path /path/to/wrds.zip
python -u run.py --task_name calibrate --market US_SP500
python -u run.py --task_name test --market US_SP500 --seed 1234
python -u run.py --task_name test --market US_SP500 --seed 4321
python -u run.py --task_name test --market US_SP500 --seed 7777
python -u run.py --task_name report --market US_SP500
```

## Protocol

Calibration decision years are 2004-2009. Their five-year evaluation windows end in 2014. Test decision years are 2015-2020 and their five-year evaluation windows end in 2025.

MVECF follows Equations 9 and 10 with `lambda_mv=10` and `gamma_mv=3`. MVECF remains an item-only baseline. Its paper-style portfolio evaluation uses equal weights because it does not output allocation weights.

The service-style experiment uses Top-5 recommendations, a 5% buy sleeve, a 5% self-financing trim sleeve, monthly rebalancing, and one-way transaction-cost stress tests at 0, 10, 25, and 50 basis points.

The US market uses FF12 preference groups when SIC coverage is sufficient. The Korean, Japanese, and German markets use twelve clusters of in-sample return-factor loadings for semi-synthetic preference groups.

The DAX paper-style audit uses Top-10 because its eligible non-holding universe is too small for Top-20.

## Output

```text
results/<MARKET>/final/
├── publication_main_table_25bps.csv
├── paper_faithful_mvecf_table.csv
├── ablation_table.csv
├── cost_sensitivity_table.csv
├── paired_year_bootstrap.csv
├── representative_orders.csv
├── representative_profile_summary.csv
├── selected_rule.csv
├── test_summary.csv
├── year_breakdown.csv
├── gamma_breakdown.csv
└── figures/
```
