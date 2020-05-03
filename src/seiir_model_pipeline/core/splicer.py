import pandas as pd
import numpy as np

from seiir_model_pipeline.core.utils import get_location_name_from_id
from seiir_model_pipeline.core.versioner import INFECTION_COL_DICT


IFR_TOL = 1e-7
COL_OBSERVED = 'observed'
COL_R_EFF = 'R_eff'
COL_BETA = 'beta'
COL_S = 'S'


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
        # differences = ratios - ratios[np.isfinite(ratios)].mean()
        # if not (differences[~differences.isnull()] < IFR_TOL).all():
        #     raise DissimilarRatioError

        ratio = ratios[np.isfinite(ratios)].mean()
        return ratio

    def concatenate_components(self, component_fit, component_forecasts):
        component_cols = [
            self.col_date, COL_BETA, COL_S
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
        return (df[COL_BETA] * df[COL_S]) / (avg_gammas * pop)

    def record_splice(self, df, col_data, observed, draw_id):
        spl = df[[self.col_date, col_data]].copy()
        spl[COL_OBSERVED] = 0.
        spl.loc[observed] = 1.
        spl[f'draw_{draw_id}'] = draw_id
        return spl

    def splice_infections(self, infection_data, i_obs, component_fit, component_forecasts):
        dates = infection_data[self.col_date]
        observations = infection_data[[self.col_date, self.col_cases]][i_obs]
        forecast_dates = dates[~i_obs]
        components = self.concatenate_components(
            component_fit=component_fit,
            component_forecasts=component_forecasts
        )
        forecasts = components.loc[components[self.col_date].isin(forecast_dates)].copy()
        df = pd.concat([observations, forecasts]).reset_index(drop=True)
        return df

    def splice_deaths(self, df, infection_data, d_obs):
        infections = infection_data[self.col_cases]
        deaths = infection_data[self.col_deaths]

        lag = self.get_lag(infection_data)
        ratio = self.get_ifr(deaths=deaths, infections=infections, lag=lag)

        df[self.col_deaths] = (df[self.col_cases] * ratio).shift(lag)
        df.loc[d_obs, self.col_deaths] = infection_data[self.col_deaths][d_obs]
        return df

    def splice_draw(self, infection_data, component_fit, component_forecasts, params, draw_id):
        pop = self.get_population(infection_data)
        import pdb; pdb.set_trace()
        i_obs = infection_data[self.col_obs_cases].astype(bool)
        d_obs = infection_data[self.col_obs_deaths].astype(bool)

        spliced = self.splice_infections(
            infection_data=infection_data, i_obs=i_obs,
            component_fit=component_fit, component_forecasts=component_forecasts
        )
        spliced = self.splice_deaths(
            df=spliced,
            infection_data=infection_data, d_obs=d_obs
        )
        spliced[COL_R_EFF] = self.compute_effective_r(df=spliced, params=params, pop=pop)

        self.infections[draw_id] = self.record_splice(
            df=spliced, col_data=self.col_cases, observed=i_obs, draw_id=draw_id
        )
        self.deaths[draw_id] = self.record_splice(
            df=spliced, col_data=self.col_deaths, observed=d_obs, draw_id=draw_id
        )
        self.reff[draw_id] = self.record_splice(
            df=spliced, col_data=COL_R_EFF, observed=d_obs, draw_id=draw_id
        )

    def format_draws(self, dictionary, id_cols, value):
        df = pd.concat(dictionary.values()).reset_index()
        wide = df.pivot(index=id_cols, columns=self.draw_cols, values=value).reset_index()
        return wide

    def save_cases(self, path):
        df = self.format_draws(self.infections, id_cols=[self.col_date, COL_OBSERVED], value=self.col_cases)
        df.to_csv(path, index=False)

    def save_deaths(self, path):
        df = self.format_draws(self.deaths, id_cols=[self.col_date, COL_OBSERVED], value=self.col_deaths)
        df.to_csv(path, index=False)

    def save_reff(self, path):
        df = self.format_draws(self.reff, id_cols=[self.col_date, COL_OBSERVED], value=COL_R_EFF)
        df.to_csv(path, index=False)
