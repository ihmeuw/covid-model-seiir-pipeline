from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Union

import numpy as np
from odeopt.ode import RK4
from odeopt.ode import ODESys
import pandas as pd


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
        params = ['beta', 'theta']

        # create component names
        components = ['S', 'E', 'I1', 'I2', 'R']

        super().__init__(self.system, params, components, *args)

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

    def __init__(self, model_specs: SeiirModelSpecs, init_cond: np.ndarray):
        self.model_specs = model_specs
        self.init_cond = init_cond

    def get_solution(self, times, beta, theta=None, solver="RK4"):
        model = CustomizedSEIIR(**asdict(self.model_specs))
        if solver == "RK4":
            solver = RK4(model.system, self.model_specs.delta)
        else:
            raise ValueError("Unknown solver type")

        if theta is None:
            theta = np.zeros(beta.size)
        else:
            assert beta.size == theta.size, f"beta ({beta.size}) and theta ({theta.size}) must have same size."

        solution = solver.solve(t=times, init_cond=self.init_cond, t_params=times,
                                params=np.vstack((beta, theta)))
        result = pd.DataFrame(
            data=np.concatenate([solution, times.reshape((1, -1)),
                                 beta.reshape((1, -1)), theta.reshape((1, -1))], axis=0).T,
            columns=["S", "E", "I1", "I2", "R", "t", "beta", "theta"]
        )
        return result


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
                       .loc[observed, ['location_id', 'date', 'deaths_draw']]
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
