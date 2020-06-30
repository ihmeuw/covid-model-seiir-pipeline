from functools import reduce
from pathlib import Path
from typing import List, Iterable

import pandas as pd
import yaml

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification


class RegressionDataInterface:

    def __init__(self,
                 regression_paths: paths.RegressionPaths,
                 ode_paths: paths.ODEPaths,
                 covariate_paths: paths.CovariatePaths):
        self.regression_paths = regression_paths
        self.ode_paths = ode_paths
        self.covariate_paths = covariate_paths

    @classmethod
    def from_specification(cls, specification: RegressionSpecification) -> 'RegressionDataInterface':
        regression_paths = paths.RegressionPaths(Path(specification.data.output_root), read_only=False)
        ode_paths = paths.ODEPaths(Path(specification.data.ode_fit_version))
        covariate_paths = paths.CovariatePaths(Path(specification.data.covariate_version))
        return cls(
            regression_paths=regression_paths,
            ode_paths=ode_paths,
            covariate_paths=covariate_paths
        )

    #####################
    # ODE paths loaders #
    #####################

    def load_location_ids(self) -> List[int]:
        with self.ode_paths.location_metadata.open() as location_file:
            location_ids = yaml.full_load(location_file)
        return location_ids

    def load_ode_fits(self, draw_id: int, location_ids: List[int]) -> pd.DataFrame:
        df_beta = pd.read_csv(self.ode_paths.get_beta_fit_file(draw_id))
        df_beta = df_beta[df_beta['location_id'].isin(location_ids)]
        df_beta['date'] = pd.to_datetime(df_beta['date'])
        return df_beta

    def get_draw_count(self) -> int:
        with self.ode_paths.fit_specification.open() as fit_spec_file:
            fit_spec = yaml.full_load(fit_spec_file)
        return fit_spec['parameters']['n_draws']

    ###########################
    # Covariate paths loaders #
    ###########################

    def check_covariates(self, covariates: Iterable[str]):
        """Ensure a reference scenario exists for all covariates.

        The reference scenario file is used to find the covariate values
        in the past (which we'll use to perform the regression).

        """
        missing = []

        for covariate in covariates:
            if covariate != 'intercept':
                covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, 'reference')
                if not covariate_path.exists():
                    missing.append(covariate)

        if missing:
            raise ValueError('All covariates supplied in the regression specification'
                             'must have a reference scenario in the covariate pool. Covariates'
                             f'missing a reference scenario: {missing}.')

    def load_covariate(self, covariate: str, location_ids: List[int]) -> pd.DataFrame:
        covariate_path = self.covariate_paths.get_covariate_scenario_file(covariate, 'reference')
        covariate_df = pd.read_csv(covariate_path)
        index_columns = ['location_id']
        covariate_df = covariate_df.loc[covariate_df['location_id'].isin(location_ids), :]
        if 'date' in covariate_df.columns:
            covariate_df['date'] = pd.to_datetime(covariate_df['date'])
            index_columns.append('date')
        covariate_df = covariate_df.rename(columns={f'{covariate}_reference': covariate})
        return covariate_df.loc[:, index_columns + [covariate]].set_index(index_columns)

    def load_covariates(self, covariates: Iterable[str], location_ids: List[int]) -> pd.DataFrame:
        covariate_data = []
        for covariate in covariates:
            if covariate != 'intercept':
                covariate_data.append(self.load_covariate(covariate, location_ids))
        covariate_data = reduce(lambda x, y: x.merge(y, left_index=True, right_index=True), covariate_data)
        return covariate_data.reset_index()

    ############################
    # Regression paths writers #
    ############################

    def save_regression_coefficients(self, coefficients: pd.DataFrame, draw_id: int) -> None:
        coefficients.to_csv(self.regression_paths.get_coefficient_file(draw_id))

    def save_regression_betas(self, df: pd.DataFrame, draw_id: int) -> None:
        beta_file = self.regression_paths.get_beta_regression_file(draw_id)
        df.to_csv(beta_file, index=False)
