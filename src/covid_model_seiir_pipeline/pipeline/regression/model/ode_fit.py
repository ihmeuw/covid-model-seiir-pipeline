from loguru import logger
import numba
import numpy as np
from odeopt.ode import RK4
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
)


def run_beta_fit(df, alpha, sigma, gamma1, gamma2, day_shift,
                 solver_class=RK4, solver_dt=1e-1):

    loc_id = df['location_id'].values[0]
    N = df['population'].values[0]

    df = df.loc[df['observed_infections'] == 1].sort_values('date')
    today = df['date'].max()
    end_date = today - pd.Timedelta(days=day_shift)
    df = df.loc[df['date'] <= end_date]

    infections_threshold = 50.0
    start_date = df.loc[infections_threshold <= df['infections_draw'], 'date'].min()
    while len(df[start_date <= df['date']]) <= 2:
        infections_threshold *= 0.5
        logger.debug(f'Reduce infections threshold for {loc_id} to {infections_threshold}')
        start_date = df.loc[infections_threshold <= df['infections_draw'], 'date'].min()
        if infections_threshold < 1e-6:
            break
    df = df.loc[start_date <= df['date']].copy()

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    # parse input
    date = df['date'].values
    t = (df['date'] - df['date'].min()).dt.days.values
    obs = df['infections_draw'].values

    S = math.solve_ode(
        system=linear_first_order,
        t=t,
        init_cond=np.array([N - obs[0] - (obs[0] / 5.0) ** (1.0 / alpha)]),
        params=np.vstack([
            [0.] * len(t),
            -obs,
        ]),
        dt=solver_dt,
    ).ravel()

    E = math.solve_ode(
        system=linear_first_order,
        t=t,
        init_cond=np.array([obs[0]]),
        params=np.vstack([
            [sigma] * len(t),
            obs,
        ]),
        dt=solver_dt,
    ).ravel()

    I1 = math.solve_ode(
        system=linear_first_order,
        t=t,
        init_cond=np.array([(obs[0] / 5.0) ** (1.0 / alpha)]),
        params=np.vstack([
            [gamma1] * len(t),
            sigma * E,
        ]),
        dt=solver_dt,
    ).ravel()

    I2 = math.solve_ode(
        system=linear_first_order,
        t=t,
        init_cond=np.array([0.]),
        params=np.vstack([
            [gamma2] * len(t),
            gamma1 * I1,
        ]),
        dt=solver_dt,
    ).ravel()

    R = math.solve_ode(
        system=linear_first_order,
        t=t,
        init_cond=np.array([0.]),
        params=np.vstack([
            [0.] * len(t),
            gamma2 * I2,
        ]),
        dt=solver_dt,
    ).ravel()

    neg_S_idx = S < 0.0

    if np.any(neg_S_idx):
        id_min = np.min(np.arange(S.size)[neg_S_idx])
        S[id_min:] = S[id_min - 1]
        E[id_min:] = E[id_min - 1]
        I1[id_min:] = I1[id_min - 1]
        I2[id_min:] = I2[id_min - 1]
        R[id_min:] = R[id_min - 1]

    components = {
        'S': S,
        'E': E,
        'I1': I1,
        'I2': I2,
        'R': R
    }

    # get beta
    params = (obs / ((S / N) * (I1 + I2)**alpha))[None, :]

    beta_fit = pd.DataFrame({
        'location_id': loc_id,
        'date': date,
        'days': t,
        'beta': params[0],
        **components
    })

    dates = pd.DataFrame({
        'location_id': loc_id,
        'start_date': start_date,
        'end_date': end_date
    }, index=[0])

    return beta_fit, dates


@numba.njit
def linear_first_order(t: float, y: np.ndarray, p: np.ndarray):
    c, f = p
    x = y[0]
    dx = -c * x + f
    return np.array([dx])
