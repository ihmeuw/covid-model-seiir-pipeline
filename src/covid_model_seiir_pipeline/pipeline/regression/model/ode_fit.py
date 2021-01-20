from loguru import logger
import numpy as np
from odeopt.ode import RK4
from odeopt.ode import LinearFirstOrder
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    math,
)


class ODEProcess:

    def __init__(self, df,
                 alpha, sigma, gamma1, gamma2, day_shift,
                 solver_class=RK4,
                 solver_dt=1e-1):

        self.loc_id = df['location_id'].values[0]
        self.N = df['population'].values[0]
        self.lag_days = df['duration'].values[0]

        # ODE parameters
        self.alpha = alpha
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.day_shift = day_shift

        df = df.loc[df['observed_infections'] == 1].sort_values('date')
        today = df['date'].max()
        end_date = today - pd.Timedelta(days=day_shift)
        df = df.loc[df['date'] <= end_date]

        cases_threshold = 50.0
        start_date = df.loc[cases_threshold <= df['infections_draw'], 'date'].min()
        while len(df[start_date <= df['date']]) <= 2:
            cases_threshold *= 0.5
            logger.debug(f'Reduce cases threshold for {self.loc_id} to {cases_threshold}')
            start_date = df.loc[cases_threshold <= df['infections_draw'], 'date'].min()
            if cases_threshold < 1e-6:
                break
        df = df.loc[start_date <= df['date']].copy()

        self.start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        self.end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        # parse input
        self.date = df['date'].values
        self.t = df['date'].diff(df['date'].min()).dt.days.values
        self.obs = df['infections_draw'].values

        self.init_cond = {
            'S': self.N - self.obs[0] - 1.0,
            'E': self.obs[0],
            'I1': 1.0,
            'I2': 0.0,
            'R': 0.0
        }

        # ode solver setup
        self.solver_dt = solver_dt
        self.t_params = np.arange(np.min(self.t), np.max(self.t) + solver_dt, solver_dt)
        self.params = None
        self.components = None
        self.step_ode_sys = LinearFirstOrder(
            self.sigma,
            solver_class=solver_class,
            solver_dt=solver_dt
        )

    def process(self):
        """Process the data.
        """
        self.rhs_newE = math.linear_interpolate(self.t_params, self.t, self.obs)
        # fit the E
        self.step_ode_sys.update_given_params(c=self.sigma)
        E = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['E']]),
                                       self.t_params,
                                       self.rhs_newE[None, :])[0]

        # fit I1
        self.step_ode_sys.update_given_params(c=self.gamma1)
        # modify initial condition of I1
        self.init_cond.update({
            'I1': (self.rhs_newE[0]/5.0)**(1.0/self.alpha)
        })
        I1 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I1']]),
                                        self.t_params,
                                        self.sigma*E[None, :])[0]

        # fit I2
        self.step_ode_sys.update_given_params(c=self.gamma2)
        I2 = self.step_ode_sys.simulate(self.t_params,
                                        np.array([self.init_cond['I2']]),
                                        self.t_params,
                                        self.gamma1*I1[None, :])[0]

        # fit S
        self.init_cond.update({
            'S': self.N - self.init_cond['E'] - self.init_cond['I1']
        })
        self.step_ode_sys.update_given_params(c=0.0)
        S = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['S']]),
                                       self.t_params,
                                       -self.rhs_newE[None, :])[0]
        neg_S_idx = S < 0.0

        # fit R
        self.step_ode_sys.update_given_params(c=0.0)
        R = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['R']]),
                                       self.t_params,
                                       self.gamma2*I2[None, :])[0]

        if np.any(neg_S_idx):
            id_min = np.min(np.arange(S.size)[neg_S_idx])
            S[id_min:] = S[id_min - 1]
            E[id_min:] = E[id_min - 1]
            I1[id_min:] = I1[id_min - 1]
            I2[id_min:] = I2[id_min - 1]
            R[id_min:] = R[id_min - 1]

        self.components = {
            'S': S,
            'E': E,
            'I1': I1,
            'I2': I2,
            'R': R
        }

        # get beta
        self.params = (self.rhs_newE / ((S / self.N) * (I1 + I2)**self.alpha))[None, :]

        return self.create_result_df(), self.create_start_end_date_df()

    def predict(self, t):
        params = math.linear_interpolate(t, self.t_params, self.params)
        components = {
            'newE': math.linear_interpolate(t, self.t_params, self.rhs_newE),
            'newE_obs': math.linear_interpolate(t, self.t, self.obs),
            **{c: math.linear_interpolate(t, self.t_params, self.components[c])
               for c in self.components},
        }
        return params, components

    def create_result_df(self):
        """Create result DataFrame.
        """
        params, components = self.predict(self.t)
        df_result = pd.DataFrame({
            'location_id': self.loc_id,
            'date': self.date,
            'days': self.t,
            'beta': params[0]
        })

        for k, v in components.items():
            df_result[k] = v

        return df_result

    def create_start_end_date_df(self):
        df_result = pd.DataFrame({
            'location_id': self.loc_id,
            'start_date': self.start_date,
            'end_date': self.end_date
        }, index=[0])
        return df_result

