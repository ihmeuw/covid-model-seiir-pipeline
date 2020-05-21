from pathlib import Path
from typing import List

import pandas as pd

from seiir_model_pipeline.paths import CovariatePaths
from seiir_model_pipeline.regression.globals import COVARIATE_COL_DICT
from seiir_model_pipeline.regression import RegressionSpecification, CovariateSpecification


class CovariatePool:
    """
    Formats covariates by pulling in the files from the seir-covariates input directory.
    Deals with time dependent and independent covariates.
    """

    def __init__(self, regression_specification: RegressionSpecification):
        self.regression_specification = regression_specification

        # TODO: use this to maybe subset past and future?
        self.col_observed = COVARIATE_COL_DICT['COL_OBSERVED']
        self.col_loc_id = COVARIATE_COL_DICT['COL_LOC_ID']
        self.col_date = COVARIATE_COL_DICT['COL_DATE']

    def create_covariate_pool(self, regression_parameters: RegressionParameters,
                              regression_directories: RegressionDirectories) -> None:
        if any([spec.draws for spec in self.covariate_specifications]):
            for draw_id in range(regression_parameters.n_draws):
                df = self.generate_covariate_df(draw_id=draw_id)
                file = COVARIATES_DRAW_FILE.format(draw_id=draw_id)
                df.to_csv(self.regression_directories.regression_covariate_dir / file,
                          index=False)
        else:
            df = self.generate_covariate_df()
            df.to_csv(self.regression_directories.regression_covariate_dir / COVARIATES_FILE,
                      index=False)

    def generate_covariate_df(self, draw_id=None) -> pd.DataFrame:
        """generates a cached covariate file for a single draw or the mean"""
        dfs = pd.DataFrame()
        value_columns = []
        for spec in self.covariate_specifications:
            if spec.name == 'intercept':
                continue

            # TODO: this should be cached when we run with draws so we don't read it every loop
            df = pd.read_csv(self.covariate_dir / f"{spec.name}.csv")

            # pull out a single draw or else use the name of the cov as the draw
            if draw_id is not None:
                if spec.draws:
                    value_column = f'draw_{draw_id}'
                else:
                    value_column = spec.name
            else:
                value_column = spec.name
            value_columns.append(value_column)
            df = df.loc[~df[value_column].isnull()].copy()
            if dfs.empty:
                dfs = df
            else:
                # time dependent covariates versus not
                if self.col_date in df.columns:
                    dfs = dfs.merge(df, on=[self.col_loc_id, self.col_date])
                else:
                    dfs = dfs.merge(df, on=[self.col_loc_id])
        dfs = dfs[[self.col_loc_id, self.col_date] + value_columns]
        dfs = dfs.loc[dfs[self.col_loc_id].isin(self.location_ids)].copy()
        return dfs
