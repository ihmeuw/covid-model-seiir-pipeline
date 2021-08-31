import numba
import numpy as np

from covid_model_seiir_pipeline.lib import (
    math,
)

SOLVER_DT = 0.1


def solve_dde(system,
              t,
              init_cond,
              params,
              dist_params,
              dt=SOLVER_DT):
    t_solve = np.arange(np.min(t), np.max(t) + dt, dt / 2)
    y_solve = np.zeros((init_cond.size, t_solve.size), dtype=init_cond.dtype)
    y_solve[:, 0] = init_cond

    params = math.linear_interpolate(t_solve, t, params)
    dist_params = np.vstack([math.sample_dist(d, t_solve) for d in dist_params])

    y_solve = rk45_dde(system, t_solve, y_solve, params, dist_params, dt)
    y_solve = math.linear_interpolate(t, t_solve, y_solve)

    return y_solve, t_solve, dist_params


@numba.njit
def rk45_dde(system,
             t_solve,
             y_solve,
             params,
             dist_params,
             dt):
    grouped_dist_params = np.zeros_like(dist_params)
    grouped_dist_params[:, ::2] = (dist_params[:, ::2] + dist_params[:, 1::2]) / dt
    for i in range(2, t_solve.size, 2):
        y_past = y_solve[:, :(i - 1):2]
        dist_past = grouped_dist_params[:, :(i - 1):2]

        k1 = system(
            t_solve[i - 2],
            y_past[:, -1],
            y_past[:, :-1],
            params[:, i - 2],
            dist_past,
        )

        y_half_step = (y_solve[:, i - 2] + dt / 2 * k1)
        dist_half_step = dist_params[:, np.array([i - 1])] / dt

        k2 = system(
            t_solve[i - 1],
            y_half_step,
            y_past,
            params[:, i - 1],
            np.hstack((dist_past, dist_half_step)),
        )

        y_half_step = (y_solve[:, i - 2] + dt / 2 * k2)

        k3 = system(
            t_solve[i - 1],
            y_half_step,
            y_past,
            params[:, i - 1],
            np.hstack((dist_past, dist_half_step)),
        )

        y_full_step = (y_solve[:, i - 2] + dt * k3)
        dist_full_step = grouped_dist_params[:, np.array([i])] / dt

        k4 = system(
            t_solve[i],
            y_full_step,
            y_past,
            params[:, i],
            np.hstack((dist_past, dist_full_step)),
        )
        y_solve[:, i] = y_solve[:, i - 2] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return y_solve
