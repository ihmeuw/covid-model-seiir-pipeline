import functools
import multiprocessing
from typing import Dict, List
from pathlib import Path

import click
import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.forecasting.specification import (
    ForecastSpecification,
    FORECAST_JOBS,
)
from covid_model_seiir_pipeline.pipeline.forecasting.data import ForecastDataInterface


logger = cli_tools.task_performance_logger


def run_compute_beta_scaling_parameters(forecast_version: str, scenario: str):
    """Pre-compute the parameters for rescaling predicted beta and write out.

    The predicted beta has two issues we're attempting to adjust.

    The first issue is that the predicted beta does not share the same
    y-intercept as the beta we fit from the ODE step. To fix this,
    we compute an initial scale factor `scale_init` that shifts the whole
    time series so that it lines up with the fit beta at the day of transition.

    The second issue is the long-term scaling of the predicted beta. If we use
    our initial scaling computed from the transition day, we bias the long
    range forecast heavily towards whatever unexplained variance appears
    in the regression on the transition day. Instead we use an average of
    the residual from the regression over some period of time in the past as
    the long range scaling.

    For locations with a large number of deaths, the average of the residual
    mean across draws represents concrete unexplained variance that is biased
    in a particular direction. For locations with a small number of deaths,
    the average of the residual mean is susceptible to a lot of noise week to
    week and so we re-center the distribution of the means about zero. For
    locations in between, we linearly scale this shift between the distribution
    of residual means centered  around zero and the distribution of
    residual means centered around the actual average residual mean.
    We call this final shift `scale_final`

    Parameters
    ----------
    forecast_version
        The path to the forecast version to run this process for.
    scenario
        Which scenario in the forecast version to run this process for.

    Notes
    -----
    The last step of average residual re-centering requires information
    from all draws.  This is the only reason this process exists as separate
    from the main forecasting process.

    """
    logger.info(f"Computing beta scaling parameters for forecast "
                f"version {forecast_version} and scenario {scenario}.", context='setup')

    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    num_cores = forecast_spec.workflow.task_specifications[FORECAST_JOBS.scaling].num_cores
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    logger.info('Loading input data.', context='read')
    total_deaths = data_interface.load_total_deaths()
    beta_scaling = forecast_spec.scenarios[scenario].beta_scaling

    logger.info('Computing scaling parameters.', context='compute')
    scaling_data = compute_initial_beta_scaling_parameters(total_deaths, beta_scaling, data_interface, num_cores)
    residual_mean_offset = compute_residual_mean_offset(scaling_data, beta_scaling, total_deaths)

    logger.info('Writing scaling parameters to disk.', context='write')
    write_out_beta_scale(scaling_data, residual_mean_offset, scenario, data_interface, num_cores)

    logger.report()


def compute_residual_mean_offset(scaling_data: List[pd.DataFrame],
                                 beta_scaling: Dict,
                                 total_deaths: pd.Series) -> pd.Series:
    """Calculates the final scaling factor offset based on total deaths.

    The offset is used to totally or partially recenter the residual average
    of beta around zero based on the total number of deaths in a location.

    Parameters
    ----------
    scaling_data
        A list with a dataframe per draw being modeled. Each dataframe has
        the draw level mean of the residuals of log beta from the
        regression over a time period in the past.
    beta_scaling
        A set of parameters for the beta scaling computation. For this function
        the important parameters are the bounds on a small number of deaths,
        `offset_deaths_lower`, below which the distribution will be centered
        around zero, and a large number of deaths, `offset_deaths_upper`,
        above which the distribution of wil not be altered. In between,
        the distribution is partially re-centered.
    total_deaths
        Total number of deaths by location at the latest date observed.

    Returns
    -------
        A series with the computed offset by location.

    """
    average_log_beta_residual_mean = (pd.concat([d.log_beta_residual_mean for d in scaling_data])
                                      .groupby(level='location_id')
                                      .mean())
    deaths_lower, deaths_upper = beta_scaling['offset_deaths_lower'], beta_scaling['offset_deaths_upper']

    scaled_offset = (deaths_lower <= total_deaths) & (total_deaths < deaths_upper)
    full_offset = total_deaths < deaths_lower

    offset = pd.Series(0, index=total_deaths.index, name='log_beta_residual_mean_offset')
    scale_factor = (deaths_upper - total_deaths) / (deaths_upper - deaths_lower)
    offset.loc[scaled_offset] = scale_factor[scaled_offset] * average_log_beta_residual_mean[scaled_offset]
    offset.loc[full_offset] = average_log_beta_residual_mean[full_offset]
    return offset


def compute_initial_beta_scaling_parameters(total_deaths: pd.Series,
                                            beta_scaling: dict,
                                            data_interface: ForecastDataInterface,
                                            num_cores: int) -> List[pd.DataFrame]:
    # Serialization is our bottleneck, so we parallelize draw level data
    # ingestion and computation across multiple processes.
    _runner = functools.partial(
        compute_initial_beta_scaling_parameters_by_draw,
        total_deaths=total_deaths,
        beta_scaling=beta_scaling,
        data_interface=data_interface
    )
    _runner(0)
    draws = list(range(data_interface.get_n_draws()))
    with multiprocessing.Pool(num_cores) as pool:
        scaling_data = list(pool.imap(_runner, draws))
    return scaling_data


def compute_initial_beta_scaling_parameters_by_draw(draw_id: int,
                                                    total_deaths: pd.Series,
                                                    beta_scaling: Dict,
                                                    data_interface: ForecastDataInterface) -> pd.DataFrame:
    # Construct a list of pandas Series indexed by location and named
    # as their column will be in the output dataframe. We'll append
    # to this list as we construct the parameters.
    draw_data = [total_deaths.copy(),
                 pd.Series(beta_scaling['window_size'], index=total_deaths.index, name='window_size')]

    # Today in the data is unique by draw.  It's based on the number of tail
    # days we use from the infections elastispliner.
    transition_date = data_interface.load_transition_date(draw_id)
    import pdb; pdb.set_trace()
    beta_regression_df = data_interface.load_beta_regression(draw_id).reset_index(level='date')
    idx = beta_regression_df.index

    # Select out the transition day to compute the initial scaling parameter.
    beta_transition = beta_regression_df.loc[beta_regression_df['date'] == transition_date.loc[idx]]
    draw_data.append(beta_transition['beta'].rename('fit_final'))
    draw_data.append(beta_transition['beta_pred'].rename('pred_start'))
    draw_data.append((beta_transition['beta'] / beta_transition['beta_pred']).rename('scale_init'))

    # Compute the beta residual mean for our parameterization and hang on
    # to some ancillary information that may be useful for plotting/debugging.
    rs = np.random.RandomState(draw_id)

    a = rs.randint(1, beta_scaling['average_over_min'])
    b = rs.randint(a + 21, beta_scaling['average_over_max'])

    draw_data.append(pd.Series(a, index=total_deaths.index, name='history_days_start'))
    draw_data.append(pd.Series(b, index=total_deaths.index, name='history_days_end'))

    beta_past = (beta_regression_df
                 .loc[beta_regression_df['date'] <= transition_date.loc[idx]]
                 .reset_index()
                 .set_index(['location_id', 'date'])
                 .sort_index())

    log_beta_residual_mean = (np.log(beta_past['beta'] / beta_past['beta_pred'])
                              .groupby(level='location_id')
                              .apply(lambda x: x.iloc[-b: -a].mean())
                              .rename('log_beta_residual_mean'))
    draw_data.append(log_beta_residual_mean)
    draw_data.append(pd.Series(draw_id, index=total_deaths.index, name='draw'))

    return pd.concat(draw_data, axis=1)


def write_out_beta_scale(beta_scales: List[pd.DataFrame],
                         offset: pd.Series,
                         scenario: str,
                         data_interface: ForecastDataInterface,
                         num_cores: int) -> None:
    _runner = functools.partial(
        write_out_beta_scales_by_draw,
        data_interface=data_interface,
        offset=offset,
        scenario=scenario
    )
    with multiprocessing.Pool(num_cores) as pool:
        pool.map(_runner, beta_scales)


def write_out_beta_scales_by_draw(beta_scales: pd.DataFrame, data_interface: ForecastDataInterface,
                                  offset: pd.Series, scenario: str) -> None:
    # Compute these draw specific parameters now that we have the offset.
    beta_scales['log_beta_residual_mean_offset'] = offset
    beta_scales['log_beta_residual_mean'] -= offset
    beta_scales['scale_final'] = np.exp(beta_scales['log_beta_residual_mean'])
    draw_id = beta_scales['draw'].iat[0]
    data_interface.save_beta_scales(beta_scales.reset_index(), scenario, draw_id)


@click.command()
@cli_tools.with_task_forecast_version
@cli_tools.with_scenario
@cli_tools.add_verbose_and_with_debugger
def beta_residual_scaling(forecast_version: str, scenario: str,
                          verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_compute_beta_scaling_parameters, logger, with_debugger)
    run(forecast_version=forecast_version,
        scenario=scenario)


if __name__ == '__main__':
    beta_residual_scaling()
