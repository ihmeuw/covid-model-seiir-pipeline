from loguru import logger
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

    cases_threshold = 50.0
    start_date = df.loc[cases_threshold <= df['infections_draw'], 'date'].min()
    while len(df[start_date <= df['date']]) <= 2:
        cases_threshold *= 0.5
        logger.debug(f'Reduce cases threshold for {loc_id} to {cases_threshold}')
        start_date = df.loc[cases_threshold <= df['infections_draw'], 'date'].min()
        if cases_threshold < 1e-6:
            break
    df = df.loc[start_date <= df['date']].copy()

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    # parse input
    date = df['date'].values
    t = (df['date'] - df['date'].min()).dt.days.values
    obs = df['infections_draw'].values

    init_cond = {
        'S': N,
        'E': obs[0],
        'I1': 1.0,
        'I2': 0.0,
        'R': 0.0
    }

    # ode solver setup
    t_params = np.arange(np.min(t), np.max(t) + solver_dt, solver_dt)

    rhs_newE = math.linear_interpolate(t_params, t, obs)
    # fit the E
    e_sys = LinearFirstOrder(sigma, solver_class, solver_dt)
    E = e_sys.simulate(t_params,
                       np.array([init_cond['E']]),
                       t_params,
                       rhs_newE[None, :])[0]

    # fit I1
    # modify initial condition of I1
    init_cond.update({
        'I1': (rhs_newE[0]/5.0)**(1.0/alpha)
    })
    i1_sys = LinearFirstOrder(gamma1, solver_class, solver_dt)
    I1 = i1_sys.simulate(t_params,
                         np.array([init_cond['I1']]),
                         t_params,
                         sigma*E[None, :])[0]

    # fit I2
    i2_sys = LinearFirstOrder(gamma2, solver_class, solver_dt)
    I2 = i2_sys.simulate(t_params,
                         np.array([init_cond['I2']]),
                         t_params,
                         gamma1*I1[None, :])[0]

    # fit S
    init_cond.update({
        'S': N - init_cond['E'] - init_cond['I1']
    })
    s_sys = LinearFirstOrder(0., solver_class, solver_dt)
    S = s_sys.simulate(t_params,
                       np.array([init_cond['S']]),
                       t_params,
                       -rhs_newE[None, :])[0]
    neg_S_idx = S < 0.0

    # fit R
    r_sys = LinearFirstOrder(0., solver_class, solver_dt)
    R = r_sys.simulate(t_params,
                       np.array([init_cond['R']]),
                       t_params,
                       gamma2*I2[None, :])[0]

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
    params = (rhs_newE / ((S / N) * (I1 + I2)**alpha))[None, :]

    params = math.linear_interpolate(t, t_params, params)
    components = {
        'newE': math.linear_interpolate(t, t_params, rhs_newE),
        'newE_obs': math.linear_interpolate(t, t, obs),
        **{c: math.linear_interpolate(t, t_params, components[c])
           for c in components},
    }

    beta_fit = pd.DataFrame({
        'location_id': loc_id,
        'date': date,
        'days': t,
        'beta': params[0]
    })

    for k, v in components.items():
        beta_fit[k] = v

    dates = pd.DataFrame({
        'location_id': loc_id,
        'start_date': start_date,
        'end_date': end_date
    }, index=[0])

    return beta_fit, dates


class LinearFirstOrder:

    def __init__(self, c, solver_class, solver_dt):
        self.c = c
        self.solver = solver_class(self.system, solver_dt)

    def system(self, t, y, p):
        f = p[0]
        x = y[0]
        dx = -self.c*x + f
        return np.array([dx])

    def simulate(self, t, init_cond, t_params, params):
        t = np.sort(np.unique(np.array(t)))
        return self.solver.solve(t, init_cond, t_params, params)

