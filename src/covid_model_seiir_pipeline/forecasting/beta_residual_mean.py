from argparse import ArgumentParser, Namespace
from typing import Optional
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.math import compute_beta_hat
from covid_model_seiir_pipeline.forecasting import model
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface


log = logging.getLogger(__name__)


def run_beta_residual_mean(forecast_version: str, scenario_name: str):
    log.info("Initiating SEIIR beta residual mean.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)

    total_deaths = data_interface.load_total_deaths()

    beta_scaling = forecast_spec.scenarios[scenario_name].beta_scaling

    def compute_residual_average(draw_id: int):
        dates_df = data_interface.load_dates_df(draw_id)
        transition_date = dates_df.set_index('location_id').sort_index()['end_date'].rename('date')
        beta_regression_df = data_interface.load_beta_regression(draw_id)
        beta_regression_df = beta_regression_df.set_index('location_id').sort_index()
        idx = beta_regression_df.index
        beta_past = (beta_regression_df
                     .loc[beta_regression_df['date'] <= transition_date.loc[idx]]
                     .reset_index()
                     .set_index(['location_id', 'date'])
                     .sort_index())
        import pdb; pdb.set_trace()

        log_beta_resid = np.log(beta_past['beta'] / beta_past['beta_pred']).rename('beta_resid')

        rs = np.random.RandomState(draw_id)
        a = rs.randint(1, beta_scaling['average_over_min'])
        b = rs.randint(a + 7, beta_scaling['average_over_max'])

        log_beta_resid.groupby(level='location_id').apply(lambda x: x.iloc[-b:-a].mean())



    # betas, scale_params = model.beta_shift(beta_past, beta_hat, transition_date,
    #                                        draw_id, **scenario_spec.beta_scaling)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--forecast-version", type=str, required=True)
    parser.add_argument("--scenario-name", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    run_beta_residual_mean(forecast_version=args.forecast_version,
                           scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()
