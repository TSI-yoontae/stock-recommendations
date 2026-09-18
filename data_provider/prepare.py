import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_parquet(path, columns):
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        from utils.parquet import ParquetFile
        return ParquetFile(path).read(columns)


def infer_epoch_unit(values):
    x = pd.to_numeric(values, errors='coerce').to_numpy(dtype='float64')
    med = float(np.nanmedian(np.abs(x)))
    if med > 1e14:
        return 'ns'
    if med > 1e11:
        return 'ms'
    if med > 1e8:
        return 's'
    return 'D'


def prepare_market(args):
    root = Path(args.root_path)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    daily = raw / (args.market + '_daily.parquet')
    if not daily.exists():
        member = 'wrds/' + args.market + '_daily.parquet'
        with zipfile.ZipFile(args.archive_path) as zf:
            with zf.open(member) as source, open(daily, 'wb') as target:
                target.write(source.read())
    if args.market == 'US_SP500':
        columns = ['date', 'permno', 'ret', 'mktcap', 'siccd', 'ticker', 'company_name']
        df = read_parquet(daily, columns)
        df['asset_key'] = pd.to_numeric(df['permno'], errors='raise').astype('int64').astype(str)
        df['permno'] = pd.to_numeric(df['permno'], errors='raise').astype('int64')
        df['isin'] = None
    else:
        columns = ['date', 'gvkey', 'iid', 'ret_local', 'mktcap_local', 'ticker', 'company_name', 'isin']
        df = read_parquet(daily, columns)
        df['asset_key'] = df['gvkey'].astype(str) + ':' + df['iid'].astype(str)
        keys = sorted(df['asset_key'].dropna().unique())
        mapping = {key: i + 1 for i, key in enumerate(keys)}
        df['permno'] = df['asset_key'].map(mapping).astype('int64')
        df['ret'] = pd.to_numeric(df['ret_local'], errors='coerce')
        df['mktcap'] = pd.to_numeric(df['mktcap_local'], errors='coerce')
        df['siccd'] = np.nan
        df['ticker'] = df['ticker'].fillna(df['isin']).fillna(df['asset_key']).astype(str)
    if np.issubdtype(df['date'].dtype, np.number):
        df['date'] = pd.to_datetime(df['date'], unit=infer_epoch_unit(df['date']), origin='unix')
    else:
        df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df['mktcap'] = pd.to_numeric(df['mktcap'], errors='coerce')
    df['gross'] = 1 + df['ret']
    monthly = (
        df.sort_values(['month', 'permno', 'date'])
        .groupby(['month', 'permno'], as_index=False, sort=False)
        .agg(
            gross=('gross', 'prod'),
            mktcap=('mktcap', 'last'),
            siccd=('siccd', 'last'),
            ticker=('ticker', 'last'),
            company_name=('company_name', 'last'),
            isin=('isin', 'last'),
            asset_key=('asset_key', 'last'),
            n_days=('ret', 'count'),
        )
    )
    monthly['ret'] = monthly.pop('gross') - 1
    monthly.loc[monthly.n_days.eq(0), 'ret'] = np.nan
    monthly = monthly[['month', 'permno', 'ret', 'mktcap', 'siccd', 'ticker', 'company_name', 'isin', 'asset_key', 'n_days']]
    monthly = monthly.sort_values(['month', 'permno']).reset_index(drop=True)
    output = root / args.data_path
    output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_pickle(output)
    print(output)
    print(len(monthly), monthly.permno.nunique(), monthly.month.min(), monthly.month.max())
