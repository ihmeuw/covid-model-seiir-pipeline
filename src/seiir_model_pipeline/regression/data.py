from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

from seiir_model_pipeline.paths import RegressionPaths, CovariatePaths, ODEPaths
from seiir_model_pipeline.regression.specification import RegressionData
from seiir_model_pipeline.globals import COVARIATE_COL_DICT


class RegressionDataInterface:

    covariate_scenario_val = "{covariate}_{scenario}"

    def __init__(self, regression_data: RegressionData):
        self.regression_paths = RegressionPaths(Path(regression_data.output_root))
        self.covariate_paths = CovariatePaths(Path(regression_data.covariate_version))
        self.ode_paths = ODEPaths(Path(regression_data.ode_fit_version))

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
                covariate_set[regress_val_name] = regress_df
            else:
                if not cached_df.equals(regress_df):
                    raise RuntimeError(
                        "Regression data is not exactly equal between covariates in covariate "
                        f"pool. Input files are {scenario_map.values()}.")

            covariate_set[scenario_val_name] = scenario_df

        return covariate_set

    def save_covariates(self, df: pd.DataFrame, draw_id: int) -> None:
        path = self.regression_paths.get_covariates_file(draw_id)
        df.to_csv(path, index=False)

    def save_scenarios(self, df: pd.DataFrame, draw_id: int) -> None:
        path = self.regression_paths.get_scenarios_file(draw_id)
        df.to_csv(path, index=False)
