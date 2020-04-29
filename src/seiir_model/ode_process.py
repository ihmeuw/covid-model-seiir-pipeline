# -*- coding: utf-8 -*-
"""
    ODE Process
    ~~~~~~~~~~~~
"""
import numpy as np
from odeopt.ode import ODESys
from odeopt.ode import RK4
from odeopt.ode import LinearFirstOrder
from odeopt.core.utils import linear_interpolate
from .spline_fit import SplineFit


class SingleGroupODEProcess:
    def __init__(self, t, obs,
                 alpha, sigma, gamma1, gamma2, N,
                 solver_class=RK4,
                 solver_dt=1e-1,
                 spline_options=None):
        """Constructor of the SingleGroupODEProcess

        Args:
            t (np.ndarray): Time variable.
            obs (np.ndarray): Cases variable.
            alpha (float): ODE parameter.
            sigma (float): ODE parameter.
            gamma1 (float): ODE parameter.
            gamma2 (float): ODE parameter.
            N (float): ODE parameter.
            solver_class (ODESolver, optional): Solver for the ODE system.
            solver_dt (float, optional): Step size for the ODE system.
            spline_options (dict | None, optional):
                Dictionary of spline prior options.
        """
        # observations
        self.t = t
        self.obs = obs

        # ODE parameters
        self.alpha = alpha
        self.sigma = sigma
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.N = N
        self.init_cond = {
            'S': self.N,
            'E': self.obs[0],
            'I1': 1.0,
            'I2': 0.0,
            'R': 0.0
        }

        # ode solver setup
        self.solver_class = solver_class
        self.solver_dt = solver_dt
        self.t_span = np.array([np.min(self.t), np.max(self.t)])
        self.t_params = np.arange(self.t_span[0],
                                  self.t_span[1] + self.solver_dt,
                                  self.solver_dt)
        self.params = None
        self.components = None
        self.create_ode_sys()

        # spline solver setup
        self.spline_options = {
            'spline_knots': np.linspace(0.0, 1.0, 7),
            'spline_degree': 3,
            'prior_spline_convexity': 'concave',
        }
        if spline_options is not None:
            self.spline_options.update(**spline_options)
        self.create_spline()

    def create_spline(self):
        """Create spline fit object.
        """
        self.step_spline_model = SplineFit(
            self.t,
            np.log(np.maximum(1.0, self.obs)),
            self.spline_options
        )

    def create_ode_sys(self):
        """Create 1D ODE solver.
        """
        self.step_ode_sys = LinearFirstOrder(
            self.sigma,
            solver_class=self.solver_class,
            solver_dt=self.solver_dt
        )

    def update_ode_params(self,
                          alpha=None,
                          sigma=None,
                          gamma1=None,
                          gamma2=None):
        """Update given parameters.

        Args:
            alpha (float | None, optional):
                Updated alpha parameter, if `None` no update will happen.
            sigma (float | None, optional):
                Updated sigma parameter, if `None` no update will happen.
            gamma1 (float | None, optional):
                Updated gamma1 parameter, if `None` no update will happen.
            gamma2 (float | None, optional):
                Updated gamma2 parameter, if `None` no update will happen.
        """
        if alpha is not None:
            assert 0.0 < alpha <= 1.0
            self.alpha = alpha
        if sigma is not None:
            assert sigma >= 0.0
            self.sigma = sigma
        if gamma1 is not None:
            assert gamma1 >= 0.0
            self.gamma1 = gamma1
        if gamma2 is not None:
            assert gamma2 >= 0.0
            self.gamma2 = gamma2

    def update_data(self, obs):
        """Update data.

        Args:
            obs (np.ndarray): independent variable.
        """
        assert self.obs.size == obs.size
        self.obs = obs
        self.create_spline()

    def fit_spline(self):
        """Fit spline.
        """
        self.step_spline_model.fit_spline()
        rhs_newE = np.exp(self.step_spline_model.predict(self.t_params))
        rhs_newE[self.t_params > self.step_spline_model.spline.knots[-1]] = 0.0
        self.rhs_newE = rhs_newE

    def process(self, fit_spline=True):
        """Process the data.
        """
        # fit the spline and predict the right-hand-side
        if fit_spline:
            self.fit_spline()

        # fit the E
        self.step_ode_sys.update_given_params(c=self.sigma)
        E = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['E']]),
                                       self.t_params,
                                       self.rhs_newE[None, :])[0]

        # fit I1
        self.step_ode_sys.update_given_params(c=self.gamma1)
        # modify initial condition of I1
        I1_org = self.init_cond['I1']
        self.init_cond.update({
            'I1': max(I1_org, (self.rhs_newE[0]/3.0)**(1.0/self.alpha))
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
        self.step_ode_sys.update_given_params(c=0.0)
        S = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['S']]),
                                       self.t_params,
                                       -self.rhs_newE[None, :])[0]

        # fit R
        self.step_ode_sys.update_given_params(c=0.0)
        R = self.step_ode_sys.simulate(self.t_params,
                                       np.array([self.init_cond['R']]),
                                       self.t_params,
                                       self.gamma2*I2[None, :])[0]

        self.components = {
            'S': S,
            'E': E,
            'I1': I1,
            'I2': I2,
            'R': R
        }

        # get beta
        self.params = (self.rhs_newE/
                       ((S/self.N)*
                        ((I1 + I2)**self.alpha)))[None, :]

    def predict(self, t):
        params = linear_interpolate(t, self.t_params, self.params)
        components = {
            c: linear_interpolate(t, self.t_params, self.components[c])
            for c in self.components
        }
        return params, components
