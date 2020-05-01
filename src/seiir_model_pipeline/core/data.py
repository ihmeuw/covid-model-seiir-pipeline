import pandas as pd
from typing import List


def load_all_location_data(directories, location_ids):
    dfs = dict()
    for loc in location_ids:
        file = directories.get_infection_file(location_id=loc, input_dir=directories.infection_dir)
        dfs[loc] = pd.read_csv(file)
    return dfs


def load_covariates(directories, covariate_names, location_id=None, forecasted=False):
    return pd.DataFrame()


def load_mr_coefficients(directories, draw_id, location_id):
    return pd.DataFrame()


def save_mr_coefficients(directories):
    pass
