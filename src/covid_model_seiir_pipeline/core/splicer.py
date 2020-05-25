import pandas as pd
import numpy as np
from datetime import datetime

from covid_model_seiir_pipeline.core.utils import get_location_name_from_id
from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT


# Tolerance for quality checking
# infectionator IFRs
IFR_TOL = 1e-7

# Column names
COL_OBSERVED = 'observed'
COL_R_EFF = 'R_eff'
COL_BETA = 'beta'
COL_S = 'S'
COL_INFECT1 = 'I1'
COL_INFECT2 = 'I2'


class DissimilarRatioError(Exception):
    pass


class Splicer:
    def __init__(self, n_draws, location_id):

        self.n_draws = n_draws
        self.draw_cols = [f'draw_{i}' for i in range(self.n_draws)]
        self.location_id = location_id

        self.location_name = None

        self.col_loc_id = INFECTION_COL_DICT['COL_LOC_ID']
        self.col_date = INFECTION_COL_DICT['COL_DATE']

        self.col_cases = INFECTION_COL_DICT['COL_CASES']
        self.col_deaths = INFECTION_COL_DICT['COL_DEATHS']

        self.col_id_lag = INFECTION_COL_DICT['COL_ID_LAG']
        self.col_obs_deaths = INFECTION_COL_DICT['COL_OBS_DEATHS']
        self.col_obs_cases = INFECTION_COL_DICT['COL_OBS_CASES']
        self.col_deaths_data = INFECTION_COL_DICT['COL_DEATHS_DATA']

        self.col_pop = INFECTION_COL_DICT['COL_POP']

        self.infections = {}
        self.deaths = {}
        self.reff = {}

    def capture_location_name(self, metadata_path):
        self.location_name = get_location_name_from_id(self.location_id, metadata_path)

    def get_lag(self, infection_data):
        lag = infection_data[self.col_id_lag]
        lag = lag.unique()
        assert len(lag) == 1
        return lag[0]

    def get_population(self, infection_data):
        pop = infection_data[self.col_pop]
        pop = pop.unique()
        assert len(pop) == 1
        return pop[0]

    @staticmethod
    def get_ifr(deaths, infections, lag):
        # Get the IFRs
        ratios = (deaths / infections.shift(lag))

        # Quality control check
        differences = ratios - ratios[np.isfinite(ratios)].mean()
        if not (differences[~differences.isnull()] < IFR_TOL).all():
            raise DissimilarRatioError

        ratio = ratios[np.isfinite(ratios)].mean()
        return ratio

    def concatenate_components(self, component_fit, component_forecasts):
        component_cols = [
            self.col_date, COL_BETA, COL_S, COL_INFECT1, COL_INFECT2
        ]
        df = pd.concat([
            component_fit.iloc[:-1][component_cols],
            component_forecasts[component_cols]
        ]).reset_index(drop=True)

        # "Newly exposed" is our cases (or infections)
        newE = -np.diff(df['S'])
        newE = np.append([0.], newE)
        df[self.col_cases] = newE
        df = df.iloc[1:].copy()
        return df

    @staticmethod
    def compute_effective_r(df, params, pop):
        avg_gammas = 1. / (1. / params['gamma1'] + 1. / params['gamma2'])
        R_C = df[COL_BETA] * params['alpha'] * (df[COL_INFECT1] + df[COL_INFECT2]) ** (params['alpha'] - 1) / avg_gammas
        return (R_C * df[COL_S]) / pop

    def record_splice(self, df, col_data, draw_id):
        spl = df[[self.col_date, COL_OBSERVED, col_data]].copy()
        spl['draw'] = f'draw_{draw_id}'
        return spl

    def splice_infections(self, infection_data, today, component_fit, component_forecasts):
        observations = infection_data[[self.col_date, self.col_cases]]
        observations = observations.loc[pd.to_datetime(observations[self.col_date]) <= today]

        components = self.concatenate_components(
            component_fit=component_fit,
            component_forecasts=component_forecasts
        )
        observations = observations.merge(
            components[[self.col_date, COL_INFECT1, COL_INFECT2, COL_BETA, COL_S]], on=[self.col_date], how='left'
        )
        forecasts = components.loc[pd.to_datetime(components[self.col_date]) > today].copy()
        df = pd.concat([observations, forecasts]).reset_index(drop=True)
        return df

    def splice_deaths(self, df, infection_data, today):
        infections = infection_data[self.col_cases]
        deaths = infection_data[self.col_deaths]

        lag = self.get_lag(infection_data)
        ratio = self.get_ifr(deaths=deaths, infections=infections, lag=lag)

        df[self.col_deaths] = (df[self.col_cases] * ratio).shift(lag)

        df_past = pd.to_datetime(df[self.col_date]) <= today
        infect_past = pd.to_datetime(infection_data[self.col_date]) <= today

        assert sum(df_past) == sum(infect_past)

        df.loc[df_past, self.col_deaths] = infection_data[self.col_deaths_data][infect_past]

        df[COL_OBSERVED] = 0
        df.loc[df_past, COL_OBSERVED] = 1
        return df

    def get_today(self, infection_data):
        d_today = infection_data.loc[infection_data[self.col_obs_deaths] == 1, self.col_date].max()
        i_today = infection_data.loc[infection_data[self.col_obs_cases] == 1, self.col_date].max()

        d_today = np.datetime64(d_today)
        i_today = np.datetime64(i_today)

        lag = self.get_lag(infection_data)
        return d_today, i_today

    def splice_draw(self, infection_data, component_fit, component_forecasts, params, draw_id):
        """
        Main function to splice a draw.

        :param infection_data: infectionator outputs
        :param component_fit: SEIR components from the past
        :param component_forecasts: SEIR components for the future
        :param params: alpha, sigma, gamma1, gamma2 parameters -- used to compute R effective
        :param draw_id: (int)
        :return:
        """
        pop = self.get_population(infection_data)
        d_today, i_today = self.get_today(infection_data)

        spliced = self.splice_infections(
            infection_data=infection_data, today=i_today,
            component_fit=component_fit, component_forecasts=component_forecasts
        )
        spliced = self.splice_deaths(
            df=spliced,
            infection_data=infection_data, today=d_today
        )
        spliced[COL_R_EFF] = self.compute_effective_r(df=spliced, params=params, pop=pop)

        self.infections[draw_id] = self.record_splice(
            df=spliced, col_data=self.col_cases, draw_id=draw_id
        )
        self.deaths[draw_id] = self.record_splice(
            df=spliced, col_data=self.col_deaths, draw_id=draw_id
        )
        self.reff[draw_id] = self.record_splice(
            df=spliced, col_data=COL_R_EFF, draw_id=draw_id
        )

    def format_draws(self, dictionary, id_cols):
        df = pd.concat(dictionary.values()).reset_index(drop=True)
        if COL_OBSERVED not in id_cols:
            df.drop(COL_OBSERVED, axis=1, inplace=True)
        wide = df.set_index(id_cols + ['draw']).unstack().reset_index()
        wide.columns = id_cols + self.draw_cols
        wide['location'] = self.location_name
        wide['location_id'] = self.location_id
        # The "observed" column is draw-specific for infections but not for deaths.
        # So if COL_OBSERVED is not one of the ID columns (ex. for infections)
        # observed is always 0 because "observed" has no meaning for infections
        # or R effective because it's all estimated.
        if COL_OBSERVED not in id_cols:
            wide[COL_OBSERVED] = 0
        return wide[['location', 'location_id', COL_OBSERVED] + id_cols + self.draw_cols]

    def save_cases(self, path):
        df = self.format_draws(self.infections, id_cols=[self.col_date])
        df.to_csv(path, index=False)

    def save_deaths(self, path):
        df = self.format_draws(self.deaths, id_cols=[self.col_date, COL_OBSERVED])
        df.to_csv(path, index=False)

    def save_reff(self, path):
        df = self.format_draws(self.reff, id_cols=[self.col_date])
        df.to_csv(path, index=False)
