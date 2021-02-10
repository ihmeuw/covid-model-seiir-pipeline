import pandas as pd


def make_ifr(infections: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    return deaths / infections


def make_ihr(infections: pd.DataFrame, hospital_admissions: pd.DataFrame) -> pd.DataFrame:
    return hospital_admissions / infections


def make_idr(infections: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    return cases / infections
