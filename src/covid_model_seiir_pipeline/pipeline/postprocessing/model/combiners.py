import pandas as pd


def make_ifr(infections: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    return deaths / infections
