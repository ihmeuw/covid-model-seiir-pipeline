from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from seiir_model_pipeline.static_vars import COVARIATE_COL_DICT, FIT_SPECIFICATION_FILE
from seiir_model_pipeline.ode_fit import FitSpecification
from seiir_model_pipeline.paths import (RegressionPaths, CovariatePaths, ODEPaths,
                                        InfectionPaths)
from seiir_model_pipeline.regression.specification import RegressionData


class RegressionDataInterface:

    covariate_scenario_val = "{covariate}_{scenario}"

    def __init__(self, regression_data: RegressionData):
        self.regression_paths = RegressionPaths(Path(regression_data.output_root))
        self.covariate_paths = CovariatePaths(Path(regression_data.covariate_version))
        self.ode_paths = ODEPaths(Path(regression_data.ode_fit_version))

        ode_fit_spec: FitSpecification = FitSpecification.from_path(self.ode_paths.root_dir /
                                                                    FIT_SPECIFICATION_FILE)

        self.infection_paths = InfectionPaths(Path(ode_fit_spec.data.infection_version))

        # TODO: transition to using data from snapshot
        file = f'location_metadata_{ode_fit_spec.data.location_set_version_id}.csv'
        file_path = Path('/ihme/covid-19/seir-pipeline-outputs/metadata-inputs') / file
        self.location_metadata_file = file_path

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

        # is time dependent cov? filter by timeseries
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

    def load_covariate(self, covariate: str, location_ids: List[int], draw_id: int,
                       use_draws: bool
                       ) -> Dict[str, pd.DataFrame]:
        covariate_set: Dict[str, pd.DataFrame] = {}
        scenario_map = self.covariate_paths.get_covariate_scenario_to_file_mapping(covariate)
        for scenario, input_file in scenario_map.items():

            # name of scenario value column
            if use_draws:
                val_name = f"draw_{draw_id}"
            else:
                val_name = self.covariate_scenario_val.format(covariate=covariate,
                                                              scenario=scenario)

            regress_df, scenario_df = self._load_scenario_file(val_name, input_file,
                                                               location_ids, draw_id)

            # change name of scenario data
            scenario_val_name = self.covariate_scenario_val.format(covariate=covariate,
                                                                   scenario=scenario)
            scenario_df = scenario_df.rename(columns={val_name: scenario_val_name})

            # change name of regression data
            regress_val_name = self.covariate_scenario_val.format(covariate=covariate,
                                                                  scenario="regression")
            regress_df = regress_df.rename(columns={val_name: regress_val_name})

            # store regress data only once
            cached_df = covariate_set.get(regress_val_name)
            if cached_df is None:
                covariate_set[regress_val_name] = regress_df.reset_index(drop=True)
            else:
                if not cached_df.equals(regress_df.reset_index(drop=True)):
                    raise RuntimeError(
                        "Regression data is not exactly equal between covariates in covariate "
                        f"pool. Input files are {scenario_map.values()}.")

            covariate_set[scenario_val_name] = scenario_df

        return covariate_set

    def save_covariates(self, df: pd.DataFrame, draw_id: int) -> None:
        path = self.regression_paths.get_covariates_file(draw_id)
        df.to_csv(path, index=False)

    def save_scenarios(self, df: pd.DataFrame, draw_id: int) -> None:
        location_ids = df[COVARIATE_COL_DICT['COL_LOC_ID']].tolist()
        for l_id in location_ids:
            loc_scenario_df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']] == l_id]
            scenario_file = self.regression_paths.get_scenarios_file(l_id, draw_id)
            loc_scenario_df.to_csv(scenario_file, index=False)

    def load_location_ids(self) -> List[int]:
        return pd.read_csv(self.location_metadata_file)["location_id"].tolist()

    def load_infections(self, location_ids: List[int], draw_id: int
                        ) -> Dict[int, pd.DataFrame]:
        dfs = dict()
        for loc in location_ids:
            file = self.infection_paths.get_infection_file(location_id=loc, draw_id=draw_id)
            dfs[loc] = pd.read_csv(file)
        return dfs

    def load_ode_fits(self, location_ids: List[int], draw_id: int) -> pd.DataFrame:
        df_beta = []
        for l_id in location_ids:
            df_beta.append(pd.read_csv(self.ode_paths.get_draw_beta_fit_file(l_id, draw_id)))
        df_beta = pd.concat(df_beta).reset_index(drop=True)
        return df_beta

    def load_mr_coefficients(self, draw_id: int) -> pd.DataFrame:
        return pd.read_csv(self.regression_paths.get_draw_coefficient_file(draw_id))

    def save_regression_betas(self, df: pd.DataFrame, draw_id: int, location_ids: List[int]
                              ) -> None:
        for l_id in location_ids:
            loc_beta_fits = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']] == l_id]
            beta_file = self.regression_paths.get_draw_beta_regression_file(l_id, draw_id)
            loc_beta_fits.to_csv(beta_file, index=False)
