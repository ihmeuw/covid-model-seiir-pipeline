from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

from seiir_model_pipeline.paths import RegressionPaths, CovariatePaths, ODEPaths
from seiir_model_pipeline.regression.specification import RegressionData
from seiir_model_pipeline.globals import COVARIATE_COL_DICT


class RegressionDataInterface:

    def __init__(self, regression_data: RegressionData):
        self.regression_paths = RegressionPaths(Path(regression_data.output_root))
        self.covariate_paths = CovariatePaths(Path(regression_data.covariate_version))
        self.ode_paths = ODEPaths(Path(regression_data.ode_fit_version))

    def _load_covariate(self, input_file_name: str, location_ids: List[int], draw_id: int
                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        # read in covariate
        covariate_scenario_file = self.covariate_paths.get_scenario_file(input_file_name)
        df = pd.read_csv(covariate_scenario_file)

        value_column = input_file_name
        columns = [COVARIATE_COL_DICT['COL_LOC_ID'], value_column]

        # drop nulls
        df = df.loc[~df[value_column].isnull()]

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
            # TODO: check inclusive inequality sign
            observed_df = df[df.date <= df.end_date].copy()
            observed_df = observed_df[columns]

            # get what we will use in scenarios
            # TODO: intentionally duplicating data for safety
            future_df = df[df.date >= df.end_date].copy()
            future_df = future_df[columns]

        # otherwise just make 2 copies
        else:
            observed_df = df.copy()
            future_df = df

            # subset columns
            observed_df = observed_df[columns]
            future_df = future_df[columns]

        return observed_df, future_df

    def load_raw_covariate_scenario(self, input_file_name: str, output_file_name_regress: str,
                                    output_file_name_scenario: str, location_ids: List[int],
                                    draw_id: int
                                    ) -> Dict[str, pd.DataFrame]:
        observed_df, reference_df = self._load_covariate(input_file_name, location_ids,
                                                         draw_id)
        observed_val = output_file_name_regress
        observed_df = observed_df.rename(columns={input_file_name: observed_val})
        reference_val = output_file_name_scenario
        reference_df = reference_df.rename(columns={input_file_name: reference_val})
        return {observed_val: observed_df, reference_val: reference_df}

    def load_raw_covariate_group(self, covariate_group: str, scenarios: List[str],
                                 file_pattern: str, location_ids: List[int], draw_id: int
                                 ) -> Dict[str, pd.DataFrame]:
        # TODO: figure out how to deal with draw_id

        covariate_set: Dict[str, pd.DataFrame] = {}
        for scenario in scenarios:
            this_scenario = file_pattern.format(group=covariate_group, scenario=scenario)

            # get scenario data from disk
            observed_df, future_df = self._load_covariate(this_scenario, location_ids, draw_id)

            # standardize name of observed data
            observed_val = file_pattern.format(group=covariate_group, scenario="observed")
            observed_df = observed_df.rename(columns={this_scenario: observed_val})

            # store observed data only once
            observed_scenario = file_pattern.format(group=covariate_group, scenario="observed")
            cached_df = covariate_set.get(observed_scenario)
            if cached_df is None:
                covariate_set[observed_scenario] = observed_df
            else:
                if not cached_df.equals(observed_df):
                    raise RuntimeError(
                        "Observed data is not exactly equal between covariates in covariate "
                        f"pool. Scenarios are {scenarios}.")

            covariate_set[this_scenario] = future_df

        return covariate_set

    def save_covariate_set(self, covariate_set: Dict[str, pd.DataFrame],
                           draw_id: int):
        # TODO: deal with draws
        for covariate_scenario, df in covariate_set.items():
            file = self.regression_paths.get_covariate_file(covariate_scenario)
            df.to_csv(file, index=False)
