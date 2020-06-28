from functools import reduce
from pathlib import Path
from typing import List, Tuple, Iterable

import pandas as pd
import yaml

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.static_vars import COVARIATE_COL_DICT


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

    def _load_scenario_file(self, val_name: str, input_file: Path, location_ids: List[int],
                            draw_id: int
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # read in covariate
        df = pd.read_csv(str(input_file))
        columns = [COVARIATE_COL_DICT['COL_LOC_ID'], val_name]

        # drop nulls
        df = df.loc[~df[val_name].isnull()]

        # subset locations
        df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']].isin(location_ids)]

        # is time dependent cov? filter by time series
        if COVARIATE_COL_DICT['COL_DATE'] in df.columns:
            columns.append(COVARIATE_COL_DICT['COL_DATE'])

            # read in cutoffs
            # TODO: not optimized because we read in multiple covariates, but data is small
            cutoffs = pd.read_csv(self.ode_paths.get_draw_date_file(draw_id))
            cutoffs = cutoffs.rename(columns={"loc_id": COVARIATE_COL_DICT['COL_LOC_ID']})
            cutoffs = cutoffs.loc[cutoffs[COVARIATE_COL_DICT['COL_LOC_ID']].isin(location_ids)]
            df = df.merge(cutoffs, how="left")

            # get what was used for ode_fit
            # TODO: should we inclusive inequality sign?
            regress_df = df[df.date <= df.end_date].copy()
            regress_df = regress_df[columns]

            # get what we will use in scenarios
            # TODO: should we inclusive inequality sign?
            scenario_df = df[df.date >= df.end_date].copy()
            scenario_df = scenario_df[columns]

        # otherwise just make 2 copies
        else:
            regress_df = df.copy()
            scenario_df = df

            # subset columns
            regress_df = regress_df[columns]
            scenario_df = scenario_df[columns]

        return regress_df, scenario_df

    def save_covariates(self, df: pd.DataFrame, draw_id: int) -> None:
        path = self.regression_paths.get_covariates_file(draw_id)
        df.to_csv(path, index=False)

    def save_scenarios(self, df: pd.DataFrame, draw_id: int) -> None:
        scenario_file = self.regression_paths.get_scenarios_file(draw_id)
        df.to_csv(scenario_file, index=False)

    def load_mr_coefficients(self, draw_id: int) -> pd.DataFrame:
        return pd.read_csv(self.regression_paths.get_coefficient_file(draw_id))

    def save_regression_betas(self, df: pd.DataFrame, draw_id: int) -> None:
        beta_file = self.regression_paths.get_beta_regression_file(draw_id)
        df.to_csv(beta_file, index=False)

    def load_regression_betas(self, draw_id: int):
        beta_file = self.regression_paths.get_beta_regression_file(draw_id)
        return pd.read_csv(beta_file)
