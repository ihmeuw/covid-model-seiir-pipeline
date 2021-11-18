from pathlib import Path

import pandas as pd


def run_rates_model(hierarchy: pd.DataFrame, *args, **kwargs):
    version = Path('/ihme/covid-19/historical-model/2021_11_15.02')
    idr_and_cases = load_idr_and_cases(version)
    ifr_and_deaths = load_ifr_and_deaths(version)
    ihr_and_admissions = load_ihr_and_admissions(version)

    # make square over date, location
    pass


def load_idr_and_cases(version):
    idr = pd.read_parquet(version / 'pred_idr.parquet')
    idr['idr_hr'] = idr['pred_idr']
    idr['idr_lr'] = idr['pred_idr']
    idr = idr.drop(columns='pred_idr')
    cases = pd.read_parquet(version / 'cases.parquet').rename(columns={'daily_cases': 'cases'})
    data = pd.concat([idr, cases], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    return data


def load_ifr_and_deaths(version):
    ifr = pd.read_parquet(version / 'pred_ifr.parquet')
    ifr = ifr.rename(columns={'pred_ifr_lr': 'ifr_lr', 'pred_ifr_hr': 'ifr_hr'}).drop(columns='pred_ifr')
    deaths = pd.read_parquet(version / 'deaths.parquet').rename(columns={'daily_deaths': 'deaths'})
    data = pd.concat([ifr, deaths], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    return data


def load_ihr_and_admissions(version):
    ifr = pd.read_parquet(version / 'pred_ihr.parquet')
    ifr = ifr.rename(columns={'pred_ihr_lr': 'ihr_lr', 'pred_ihr_hr': 'ihr_hr'}).drop(columns='pred_ihr')
    deaths = pd.read_parquet(version / 'admissions.parquet').rename(
        columns={'daily_hospitalizations': 'admissions'})
    data = pd.concat([ifr, deaths], axis=1)
    data = data.loc[~data.isnull().any(axis=1)].sort_index()
    return data