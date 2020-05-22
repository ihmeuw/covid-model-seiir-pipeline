from pathlib import Path
from typing import List, Optional, Dict, Tuple

import pandas as pd

from seiir_model_pipeline.paths import RegressionPaths, CovariatePaths
from seiir_model_pipeline.regression import RegressionData
from seiir_model_pipeline.regression.globals import COVARIATE_COL_DICT


class CovariateDataInterface:
    def __init__(self, regression_data: RegressionData):
        self.covariate_paths = CovariatePaths(Path(regression_data.covariate_version))

    def load_raw_covariate_scenario(self, covariate: str, scenario: str,
                                    location_ids: List[int], draw_id: Optional[int] = None
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        covariate_scenario_file = self.covariate_paths.get_scenario_file(covariate, scenario)
        df = pd.read_csv(covariate_scenario_file)
        value_column = self.covariate_paths.scenario_file.format(covariate=covariate,
                                                                 scenario=scenario)
        columns = [COVARIATE_COL_DICT['COL_LOC_ID'], value_column]

        # is time dependent cov?
        if COVARIATE_COL_DICT['COL_DATE'] in df.columns:
            columns.append(COVARIATE_COL_DICT['COL_DATE'])

        # drop nulls
        df = df.loc[~df[value_column].isnull()]

        # subset locations
        df = df.loc[df[COVARIATE_COL_DICT['COL_LOC_ID']].isin(location_ids)]

        # subset out historical data
        observed_val = self.covariate_paths.scenario_file.format(covariate=covariate,
                                                                 scenario="observed")

        observed_df = df[df.COVARIATE_COL_DICT['COL_OBSERVED'] == 1]
        observed_df = observed_df[columns + [value_column]]
        observed_df = observed_df.rename(columes={value_column: observed_val})

        # subset columns
        future_df = df[df.COVARIATE_COL_DICT['COL_OBSERVED'] == 0]
        future_df = future_df[columns + [value_column]]

        return observed_df, future_df

    def load_raw_covariate_set(self, covariate: str, location_ids: List[int],
                               draw_id: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        # TODO: figure out how to deal with draw_id

        scenario_set = self.covariate_paths.get_scenario_set(covariate)

        covariate_set: Dict[str, pd.DataFrame] = {}
        for scenario in scenario_set:

            # get scenario data from disk
            observed_df, future_df = self.load_raw_covariate_scenario(covariate, scenario,
                                                                      location_ids, draw_id)

            # store observed data only once
            observed_scenario = self.covariate_paths.scenario_file.format(covariate=covariate,
                                                                          scenario="observed")
            cached_df = covariate_set.get(observed_scenario)
            if cached_df is None:
                covariate_set[observed_scenario] = cached_df
            else:
                if not cached_df.equals(observed_df):
                    raise RuntimeError(
                        "Observed data is not exactly equal between covariates in covariate "
                        f"pool. scenarios are {scenario_set}.")

            # store this scenario
            scenario_name = self.covariate_paths.scenario_file.format(covariate=covariate,
                                                                      scenario=scenario)
            covariate_set[scenario_name] = future_df

        return covariate_set


class RegressionDataInterface:

    def __init__(self, regression_data: RegressionData):
        self.regression_paths = RegressionPaths(Path(regression_data.output_root))

    def save_covariate_set(self, covariate_set: Dict[str, pd.DataFrame],
                           draw_id: Optional[int] = None):
        # TODO: deal with draws
        for covariate_scenario, df in covariate_set.items():
            file = self.regression_paths.get_covariate_file(covariate_scenario)
            df.to_csv(file, index=False)

    def get_covariate_scenarios(self, covariate_scenarios: List[str],
                                draw_id: Optional[int] = None) -> pd.DataFrame:
        dfs = pd.DataFrame()
        for covariate_scenario in covariate_scenarios:

            # read in scenario
            scenario_file = self.regression_paths.get_covariate_file(covariate_scenario)
            scenario_df = pd.read_csv(scenario_file)

            # read in observed data
            group = CovariatePaths.get_covariate_group_from_covariate(covariate_scenario)
            obs_name = CovariatePaths.scenario_file.format(covariate_group=group,
                                                           scenario="observed")
            observed_file = self.regression_paths.get_covariate_file(obs_name)
            observed_df = pd.read_csv(observed_file)

            # concat timeseries
            df = pd.concat([observed_df, scenario_df])

            # merge with other scenarios
            if dfs.empty:
                dfs = df
            else:
                # time dependent covariates versus not
                if COVARIATE_COL_DICT['COL_DATE'] in df.columns:
                    dfs = dfs.merge(df, on=[COVARIATE_COL_DICT['COL_LOC_ID'],
                                            COVARIATE_COL_DICT['COL_DATE']])
                else:
                    dfs = dfs.merge(df, on=[COVARIATE_COL_DICT['COL_LOC_ID']])

        return dfs
