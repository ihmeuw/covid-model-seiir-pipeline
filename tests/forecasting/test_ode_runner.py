import numpy as np

from covid_model_seiir_pipeline.forecasting.model import ODERunner, SiierdModelSpecs


def test_ode_runner():
    specs = SiierdModelSpecs(
        alpha=0.9,
        sigma=1.0,
        gamma1=0.3,
        gamma2=0.4,
        N=100,
    )
    # init_cond = [S, E, I1, I2, R] at t = 0
    init_cond = np.array([96, 0, 2, 2, 0], dtype=float)

    t = np.arange(0, 31, 1)
    dt = 0.1  # for the ODE solver, so it can be less than a day
    beta = 2*np.exp(-0.01*t)
    ode_runner = ODERunner(specs, init_cond, dt)
    result = ode_runner.get_solution(t, beta)
    print("Okay!")
