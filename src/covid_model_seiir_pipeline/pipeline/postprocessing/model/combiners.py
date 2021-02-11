import pandas as pd

INFECTION_TO_DEATH = 24
INFECTION_TO_ADMISSION = 11
INFECTION_TO_CASE = 11


def make_ifr(infections: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    return deaths / infections.groupby('location_id').shift(INFECTION_TO_DEATH)


def make_ihr(infections: pd.DataFrame, hospital_admissions: pd.DataFrame) -> pd.DataFrame:
    return hospital_admissions / infections.groupby('location_id').shift(INFECTION_TO_ADMISSION)


def make_idr(infections: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    return cases / infections.groupby('location_id').shift(INFECTION_TO_CASE)
