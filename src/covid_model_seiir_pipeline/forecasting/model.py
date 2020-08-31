from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Union

import numpy as np
from odeopt.ode import RK4
from odeopt.ode import ODESys
import pandas as pd

from covid_model_seiir_pipeline.math import compute_beta_hat


class CustomizedSEIIR(ODESys):
    """Customized SEIIR ODE system."""

    def __init__(self,
                 alpha: float,
                 sigma: float,
                 gamma1: float,
                 gamma2: float,
                 N: Union[int, float],
                 delta: float, *args):
        """Constructor of CustomizedSEIIR.
        """
        self.alpha = alpha
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.N = N
        self.delta = delta

        # create parameter names
        self.params = ['beta', 'theta']

        # create component names
        self.components = ['S', 'E', 'I1', 'I2', 'R']

        super().__init__(self.system, self.params, self.components, *args)

    def system(self, t: float, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """ODE System.
        """
        beta = p[0]
        theta = p[1]

        s = y[0]
        e = y[1]
        i1 = y[2]
        i2 = y[3]
        r = y[4]

        theta_plus = max(theta, 0.) * s / 1_000_000
        theta_minus = min(theta, 0.)
        theta_tilde = int(theta_plus != theta_minus)
        theta_minus_alt = (self.gamma1 - self.delta) * i1 - self.sigma * e - theta_plus
        effective_theta_minus = max(theta_minus, theta_minus_alt) * theta_tilde

        new_e = beta*(s/self.N)*(i1 + i2)**self.alpha

        ds = -new_e - theta_plus
        de = new_e - self.sigma*e
        di1 = self.sigma*e - self.gamma1*i1 + theta_plus + effective_theta_minus
        di2 = self.gamma1*i1 - self.gamma2*i2
        dr = self.gamma2*i2 - effective_theta_minus

        return np.array([ds, de, di1, di2, dr])


@dataclass(frozen=True)
class SeiirModelSpecs:
    alpha: float
    sigma: float
    gamma1: float
    gamma2: float
    N: float  # in case we want to do fractions, but not number of people
    delta: float = 0.1

    def __post_init__(self):
        assert 0 < self.alpha <= 1.0
        assert self.sigma >= 0.0
        assert self.gamma1 >= 0
        assert self.gamma2 >= 0
        assert self.N > 0
        assert self.delta > 0.0


class ODERunner:

    def __init__(self, solver_name: str, model_specs: SeiirModelSpecs):
        self.model = CustomizedSEIIR(**asdict(model_specs))
        if solver_name == "RK45":
            self.solver = RK4(self.model.system, model_specs.delta)
        else:
            raise NotImplementedError(f"Unknown solver type {solver_name}.")

    def get_solution(self, initial_condition, times, beta, theta):
        solution = self.solver.solve(
            t=times,
            init_cond=initial_condition,
            t_params=times,
            params=np.vstack((beta, theta))
        )

        result_array = np.concatenate([
            solution,
            beta.reshape((1, -1)),
            theta.reshape((1, -1)),
            times.reshape((1, -1))
        ], axis=0).T

        result = pd.DataFrame(
            data=result_array,
            columns=self.model.components + self.model.params + ['t']
        )

        return result


def run_normal_ode_model_by_location(initial_condition, beta_params, betas, thetas, location_ids, solver):
    forecasts = []
    for location_id in location_ids:
        init_cond = initial_condition.loc[location_id].values
        total_population = init_cond.sum()

        model_specs = SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=total_population,
        )
        ode_runner = ODERunner(solver, model_specs)

        loc_betas = betas.loc[location_id].sort_values('date')
        loc_days = loc_betas['date']
        loc_times = np.array((loc_days - loc_days.min()).dt.days)
        loc_betas = loc_betas['beta_pred'].values
        loc_thetas = np.repeat(thetas.get(location_id, default=0), loc_betas.size)

        forecasted_components = ode_runner.get_solution(init_cond, loc_times, loc_betas, loc_thetas)
        forecasted_components['date'] = loc_days.values
        forecasted_components['location_id'] = location_id
        forecasts.append(forecasted_components)
    forecasts = pd.concat(forecasts)
    return forecasts


def splice_components(components_past, components_forecast):
    shared_columns = ['date', 'S', 'E', 'I1', 'I2', 'R', 'beta']
    components_past = components_past[shared_columns].reset_index()
    components_forecast = components_forecast[['location_id'] + shared_columns]
    components = (pd.concat([components_past, components_forecast])
                  .sort_values(['location_id', 'date'])
                  .set_index(['location_id']))
    return components


def forecast_beta(covariates, coefficients, beta_shift_parameters):
    log_beta_hat = compute_beta_hat(covariates, coefficients)
    beta_hat = np.exp(log_beta_hat).rename('beta_pred').reset_index()

    # Rescale the predictions of beta based on the residuals from the
    # regression.
    betas = beta_shift(beta_hat, beta_shift_parameters).set_index('location_id')
    return betas


def compute_output_metrics(infection_data, components, seir_params):
    observed_infections, modeled_infections = compute_infections(infection_data, components)
    observed_deaths, modeled_deaths = compute_deaths(infection_data, modeled_infections)

    infections = observed_infections.combine_first(modeled_infections)['cases_draw'].rename('infections')
    deaths = observed_deaths.combine_first(modeled_deaths).rename(columns={'deaths_draw': 'deaths'})
    r_effective = compute_effective_r(infection_data, components, seir_params)

    return infections, deaths, r_effective


def beta_shift(beta_hat: pd.DataFrame,
               beta_scales: pd.DataFrame) -> pd.DataFrame:
    """Shift the raw predicted beta to line up with beta in the past.

    This method performs both an intercept shift and a scaling based on the
    residuals of the ode fit beta and the beta hat regression in the past.

    Parameters
    ----------
        beta_hat
            Dataframe containing the date, location_id, and beta hat in the
            future.
        beta_scales
            Dataframe containing precomputed parameters for the scaling.

    Returns
    -------
        Predicted beta, after scaling (shift).

    """
    beta_scales = beta_scales.set_index('location_id')
    beta_hat = beta_hat.sort_values(['location_id', 'date']).set_index('location_id')
    scale_init = beta_scales['scale_init']
    scale_final = beta_scales['scale_final']
    window_size = beta_scales['window_size']

    beta_final = []
    for location_id in beta_hat.index.unique():
        if window_size is not None:
            t = np.arange(len(beta_hat.loc[location_id])) / window_size.at[location_id]
            scale = scale_init.at[location_id] + (scale_final.at[location_id] - scale_init.at[location_id]) * t
            scale[(window_size.at[location_id] + 1):] = scale_final.at[location_id]
        else:
            scale = scale_init.at[location_id]
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_pred']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    return beta_final


def compute_infections(infection_data: pd.DataFrame,
                       components: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_infecs'] == 1
    observed_infections = (infection_data
                           .loc[observed, ['location_id', 'date', 'cases_draw']]
                           .set_index(['location_id', 'date'])
                           .sort_index())

    modeled_infections = (components
                          .groupby('location_id')['S']
                          .apply(lambda x: x.shift(1) - x)
                          .fillna(0)
                          .rename('cases_draw'))
    modeled_infections = pd.concat([components['date'], modeled_infections], axis=1).reset_index()
    modeled_infections = modeled_infections.set_index(['location_id', 'date']).sort_index()
    return observed_infections, modeled_infections


def compute_deaths(infection_data: pd.DataFrame,
                   modeled_infections: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    observed = infection_data['obs_deaths'] == 1
    observed_deaths = (infection_data
                       .loc[observed, ['location_id', 'date', 'deaths_mean']]
                       .rename(columns={'deaths_mean': 'deaths_draw'})
                       .set_index(['location_id', 'date'])
                       .sort_index())
    observed_deaths['observed'] = 1

    infection_death_lag = infection_data['i_d_lag'].max()

    def _compute_ifr(data):
        deaths = data['deaths_draw']
        infecs = data['cases_draw']
        return (deaths / infecs.shift(infection_death_lag)).dropna().mean()

    ifr = (infection_data
           .groupby('location_id')
           .apply(_compute_ifr))
    modeled_deaths = ((modeled_infections['cases_draw'] * ifr)
                      .shift(infection_death_lag)
                      .rename('deaths_draw')
                      .to_frame())
    modeled_deaths['observed'] = 0
    return observed_deaths, modeled_deaths


def compute_effective_r(infection_data: pd.DataFrame, components: pd.DataFrame,
                        beta_params: Dict[str, float]) -> pd.DataFrame:
    alpha, gamma1, gamma2 = beta_params['alpha'], beta_params['gamma1'], beta_params['gamma2']
    components = components.reset_index().set_index(['location_id', 'date'])
    beta, s, i1, i2 = components['beta'], components['S'], components['I1'], components['I2']
    n = infection_data.groupby('location_id')['pop'].max()
    avg_gamma = 1 / (1 / gamma1 + 1 / gamma2)
    r_controlled = beta * alpha / avg_gamma * (i1 + i2) ** (alpha - 1)
    r_effective = (r_controlled * s / n).rename('r_effective')
    return r_effective


def compute_reimposition_date(deaths, population, reimposition_threshold,
                              min_wait, last_reimposition_end_date=None) -> pd.Series:
    death_rate = deaths.reset_index(level='date').merge(population, on='location_id')
    death_rate['death_rate'] = death_rate['deaths'] / death_rate['population']

    projected = death_rate['observed'] == 0
    last_observed_date = death_rate[~projected].groupby('location_id')['date'].max()
    min_reimposition_date = (last_observed_date + min_wait)
    if last_reimposition_end_date is not None:
        min_reimposition_date.loc[last_reimposition_end_date.index] = last_reimposition_end_date + min_wait

    after_min_reimposition_date = death_rate['date'] >= min_reimposition_date.loc[death_rate.index]
    over_threshold = death_rate['death_rate'] > reimposition_threshold
    reimposition_date = (death_rate[over_threshold & after_min_reimposition_date]
                         .groupby('location_id')['date']
                         .min()
                         .rename('reimposition_date'))

    return reimposition_date


def compute_mobility_lower_bound(mobility: pd.DataFrame, mandate_effect: pd.DataFrame) -> pd.Series:
    min_observed_mobility = mobility.groupby('location_id').min().rename('min_mobility')
    max_mandate_mobility = mandate_effect.sum(axis=1).rename('min_mobility').reindex(min_observed_mobility.index,
                                                                                     fill_value=100)
    mobility_lower_bound = min_observed_mobility.where(min_observed_mobility <= max_mandate_mobility,
                                                       max_mandate_mobility)
    return mobility_lower_bound


def compute_rampup(reimposition_date: pd.Series,
                   percent_mandates: pd.DataFrame,
                   days_on: pd.Timedelta) -> pd.DataFrame:
    rampup = pd.merge(reimposition_date, percent_mandates.reset_index(level='date'), on='location_id', how='left')
    rampup['rampup'] = rampup.groupby('location_id')['percent'].apply(lambda x: x / x.max())
    rampup['first_date'] = rampup.groupby('location_id')['date'].transform('min')
    rampup['diff_date'] = rampup['reimposition_date'] - rampup['first_date']
    rampup['date'] = rampup['date'] + rampup['diff_date'] + days_on
    rampup = rampup.reset_index()[['location_id', 'date', 'rampup']]
    return rampup


def compute_new_mobility(old_mobility: pd.Series,
                         reimposition_date: pd.Series,
                         mobility_lower_bound: pd.Series,
                         percent_mandates: pd.DataFrame,
                         min_wait: pd.Timedelta,
                         days_on: pd.Timedelta) -> pd.Series:
    mobility = pd.merge(old_mobility.reset_index(level='date'), reimposition_date, how='left', on='location_id')
    mobility = mobility.merge(mobility_lower_bound, how='left', on='location_id')

    reimposes = mobility['reimposition_date'].notnull()
    dates_on = ((mobility['reimposition_date'] <= mobility['date'])
                & (mobility['date'] <= mobility['reimposition_date'] + min_wait))
    mobility['mobility_explosion'] = mobility['min_mobility'].where(reimposes & dates_on, np.nan)

    rampup = compute_rampup(reimposition_date, percent_mandates, days_on)

    mobility = mobility.merge(rampup, how='left', on=['location_id', 'date'])
    post_reimplementation = ~(mobility['mobility_explosion'].isnull() & mobility['rampup'].notnull())
    mobility['mobility_explosion'] = mobility['mobility_explosion'].where(
        post_reimplementation,
        mobility['min_mobility'] * mobility['rampup']
    )

    idx_columns = ['location_id', 'date']
    mobility = (mobility[idx_columns + ['mobility', 'mobility_explosion']]
                .set_index(idx_columns)
                .sort_index()
                .min(axis=1))
    return mobility

