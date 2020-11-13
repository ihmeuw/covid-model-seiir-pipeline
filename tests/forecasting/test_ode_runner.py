import numpy as np

from covid_model_seiir_pipeline.forecasting.model.ode_forecast import _ODERunner, _SeiirModelSpecs


def test_ode_runner():
    specs = _SeiirModelSpecs(
        alpha=0.9,
        sigma=1.0,
        gamma1=0.3,
        gamma2=0.4,
        N=100,
        delta=0.1,
    )
    model = 'normal'
    # init_cond = [S, E, I1, I2, R] at t = 0
    init_cond = np.array([96, 0, 2, 2, 0], dtype=float)

    t = np.arange(0, 31, 1)
    beta = 2*np.exp(-0.01*t)
    theta = np.zeros_like(beta)
    ode_runner = _ODERunner('RK45', model, specs)
    result = ode_runner.get_solution(init_cond, t, beta, theta)
    print("Okay!")
