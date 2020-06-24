from typing import Union
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
class SiierdModelSpecs:
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

    def __init__(self, model_specs: SiierdModelSpecs, init_cond: np.ndarray, dt: float):
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
