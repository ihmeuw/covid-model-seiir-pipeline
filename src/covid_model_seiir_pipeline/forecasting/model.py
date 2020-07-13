from collections import defaultdict
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


def beta_shift(beta_past: pd.DataFrame,
               beta_hat: pd.DataFrame,
               transition_date: pd.DataFrame,
               draw_id: int,
               window_size: Union[int, None] = None,
               average_over_min: int = 1,
               average_over_max: int = 35) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Shift the raw predicted beta to line up with beta in the past..

    This method performs both an intercept shift and a scaling based on the
    residuals of the ode fit beta and the beta hat regression in the past.

    Parameters
    ----------
        beta_past
            Dataframe containing the date, location_id, beta fit, and beta hat
            in the past.
        beta_hat
            Dataframe containing the date, location_id, and beta hat in the
            future.
        transition_date
            Dataframe containing location id and the date of transition from
            past to prediction.
        draw_id
            Draw of data provided.  Will be used as a seed for
            a random number generator to determine the amount of beta history
            to leverage in rescaling the y-intercept for the beta prediction.
        window_size
            Window size for the transition. If `None`, Hard shift no transition.
            Default to None.
        average_over_min, average_over_max
            Min and max duration to average residuals over. The actual duration
            will be sampled uniformly from within these bounds based on the
            draw id.

    Returns
    -------
        Predicted beta, after scaling (shift) and the initial scaling.

    """
    rs = np.random.RandomState(seed=draw_id)
    avg_over = rs.randint(average_over_min, average_over_max)

    beta_past = beta_past.set_index('location_id').sort_values('date')
    beta_hat = beta_hat.set_index('location_id').sort_values('date')
    beta_fit_final = beta_past.loc[beta_past['date'] == transition_date.loc[beta_past.index], 'beta']
    beta_hat_start = beta_hat.loc[beta_hat['date'] == transition_date.loc[beta_hat.index], 'beta_pred']
    scale_init = beta_fit_final / beta_hat_start

    beta_past = beta_past.reset_index().set_index(['location_id', 'date'])
    log_beta_resid = np.log(beta_past['beta'] / beta_past['beta_pred']).rename('beta_resid')
    scale_final = np.exp(log_beta_resid
                         .groupby(level='location_id')
                         .apply(lambda x: x.iloc[-avg_over:].mean()))

    beta_final = []
    for location_id in beta_hat.index.unique():
        if window_size is not None:
            t = np.arange(len(beta_hat.loc[location_id])) / window_size
            scale = scale_init.at[location_id] + (scale_final.at[location_id] - scale_init.at[location_id]) * t
            scale[(window_size + 1):] = scale_final.at[location_id]
        else:
            scale = scale_init.at[location_id]
        loc_beta_hat = beta_hat.loc[location_id].set_index('date', append=True)['beta_pred']
        loc_beta_final = loc_beta_hat * scale
        beta_final.append(loc_beta_final)

    beta_final = pd.concat(beta_final).reset_index()

    scale_params = pd.concat([
        beta_fit_final.rename('fit_final'),
        beta_hat_start.rename('pred_start'),
        scale_final.rename('beta_ratio_mean'),
        np.log(scale_final).rename('beta_residual_mean')
    ], axis=1).reset_index()
    scale_params['window_size'] = window_size
    scale_params['history_days'] = avg_over

    return beta_final, scale_params
