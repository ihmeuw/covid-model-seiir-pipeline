import itertools
from pathlib import Path

import pandas as pd


def run_rates_model(hierarchy: pd.DataFrame, *args, **kwargs):
    most_detailed = hierarchy[hierarchy.most_detailed == 1].location_id.unique().tolist()
    version = Path('/ihme/covid-19/historical-model/2021_11_15.02')
    measure_map = {
        ('ifr', 'deaths'): load_ifr_and_deaths,
        ('ihr', 'admissions'): load_ihr_and_admissions,
        ('idr', 'cases'): load_idr_and_cases,
    }

    rates, measures, lags = [], [], []
    for (r, m), loader in measure_map.items():
        data = loader(version).reset_index()
        lag = data.lag.iloc[0]
        data = (data
                .loc[data.location_id.isin(most_detailed)]
                .set_index(['location_id', 'date'])
                .groupby('location_id')
                .apply(lambda x: x.reset_index(level='location_id', drop=True)
                                  .shift(periods=-lag, freq='D')))
        rates.append(data.loc[:, [f'{r}_lr', f'{r}_hr']])
        measures.append(data.loc[:, m])
        lags.append(lag)

    col_order = [f'{r}_{g}' for g, r in itertools.product(['lr', 'hr'], ['ifr', 'ihr', 'idr'])]
    rates = pd.concat(rates, axis=1).loc[:, col_order]

    all_locs = rates.reset_index().location_id.unique().tolist()
    dates = rates.reset_index().date
    global_date_range = pd.date_range(dates.min(), dates.max())
    square_idx = pd.MultiIndex.from_product((all_locs, global_date_range), names=['location_id', 'date']).sort_values()

    rates = rates.reindex(square_idx).sort_index()
    measures = pd.concat(measures, axis=1).reindex(square_idx).sort_index()

    return rates, measures, lags


def load_idr_and_cases(version):
    idr = pd.read_parquet(version / 'pred_idr.parquet')
    idr['idr_hr'] = idr['pred_idr']
    idr['idr_lr'] = idr['pred_idr']
    idr = idr.drop(columns='pred_idr')
    cases = pd.read_parquet(version / 'cases.parquet').rename(columns={'daily_cases': 'cases'})
    data = pd.concat([idr, cases], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    data['lag'] = 12
    return data


def load_ifr_and_deaths(version):
    ifr = pd.read_parquet(version / 'pred_ifr.parquet')
    ifr = ifr.rename(columns={'pred_ifr_lr': 'ifr_lr', 'pred_ifr_hr': 'ifr_hr'}).drop(columns='pred_ifr')
    deaths = pd.read_parquet(version / 'deaths.parquet').rename(columns={'daily_deaths': 'deaths'})
    data = pd.concat([ifr, deaths], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    data['lag'] = 26
    return data


def load_ihr_and_admissions(version):
    ifr = pd.read_parquet(version / 'pred_ihr.parquet')
    ifr = ifr.rename(columns={'pred_ihr_lr': 'ihr_lr', 'pred_ihr_hr': 'ihr_hr'}).drop(columns='pred_ihr')
    deaths = pd.read_parquet(version / 'hospitalizations.parquet').rename(
        columns={'daily_hospitalizations': 'admissions'})
    data = pd.concat([ifr, deaths], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    data['lag'] = 12
    return data