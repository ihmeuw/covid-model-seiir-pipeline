from typing import Dict, Tuple, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from odeopt.ode import RK4
from odeopt.ode import ODESys


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

    def __init__(self, model_specs: SeiirModelSpecs, init_cond: np.ndarray, dt: float):
        self.model_specs = model_specs
        self.init_cond = init_cond
        self.dt = dt

    def get_solution(self, times, beta, theta=None, solver="RK4"):
        model = CustomizedSEIIR(**asdict(self.model_specs))
        if solver == "RK4":
            solver = RK4(model.system, self.dt)
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


def get_ode_init_cond(location_id, beta_ode_fit, current_date):
    """
    Get the initial condition for the ODE.

    Args:
        location_id (init):
            Location ids.
        beta_ode_fit (str | pd.DataFrame):
            The result for the beta_ode_fit, either file or path to file.
        current_date (str | np.datetime64):
            Current date for each location we try to predict off. Either file
            or path to file.


    Returns:
         pd.DataFrame: Initial conditions by location.
    """
    # process input
    assert (static_vars.COL_GROUP in beta_ode_fit)
    assert (static_vars.COL_DATE in beta_ode_fit)
    beta_ode_fit = beta_ode_fit[beta_ode_fit[static_vars.COL_GROUP] == location_id].copy()

    if isinstance(current_date, str):
        current_date = np.datetime64(current_date)
    else:
        assert isinstance(current_date, np.datetime64)

    dt = np.abs((pd.to_datetime(beta_ode_fit[static_vars.COL_DATE]) - current_date).dt.days)
    beta_ode_fit = beta_ode_fit.iloc[np.argmin(dt)]

    col_components = static_vars.SEIIR_COMPARTMENTS
    assert all([c in beta_ode_fit for c in col_components])

    return beta_ode_fit[col_components].values.ravel()


def beta_shift(beta_fit: pd.DataFrame,
               beta_pred: np.ndarray,
               draw_id: int,
               window_size: Union[int, None] = None,
               average_over_min: int = 1,
               average_over_max: int = 35) -> Tuple[np.ndarray, Dict[str, float]]:
    """Shift the raw predicted beta to line up with beta in the past..

    This method performs both an intercept shift and a scaling based on the
    residuals of the ode fit beta and the beta hat regression in the past.

    Parameters
    ----------
        beta_fit
            Data frame contains the date and beta fit.
        beta_pred
            beta prediction.
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
    assert 'date' in beta_fit.columns, "'date' has to be in beta_fit data frame."
    assert 'beta' in beta_fit.columns, "'beta' has to be in beta_fit data frame."
    beta_fit = beta_fit.sort_values('date')
    beta_hat = beta_fit['beta_pred'].to_numpy()
    beta_fit = beta_fit['beta'].to_numpy()

    rs = np.random.RandomState(seed=draw_id)
    avg_over = rs.randint(average_over_min, average_over_max)

    beta_fit_final = beta_fit[-1]
    beta_pred_start = beta_pred[0]

    scale_init = beta_fit_final / beta_pred_start
    log_beta_resid = np.log(beta_fit / beta_hat)
    scale_final = np.exp(log_beta_resid[-avg_over:].mean())

    scale_params = {
        'window_size': window_size,
        'history_days': avg_over,
        'fit_final': beta_fit_final,
        'pred_start': beta_pred_start,
        'beta_ratio_mean': scale_final,
        'beta_residual_mean': np.log(scale_final),
    }

    if window_size is not None:
        assert isinstance(window_size, int) and window_size > 0, f"window_size={window_size} has to be a positive " \
                                                                 f"integer."
        scale = scale_init + (scale_final - scale_init)/window_size*np.arange(beta_pred.size)
        scale[(window_size + 1):] = scale_final
    else:
        scale = scale_init

    betas = beta_pred * scale

    return betas, scale_params
