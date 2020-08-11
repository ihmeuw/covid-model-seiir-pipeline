from argparse import ArgumentParser, Namespace
import multiprocessing
from typing import Optional
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.workflow import FORECAST_SCALING_CORES


log = logging.getLogger(__name__)


def run_beta_residual_mean(forecast_version: str, scenario_name: str):
    log.info("Initiating SEIIR beta residual mean.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    data_interface = ForecastDataInterface.from_specification(forecast_spec)
    locations = data_interface.load_location_ids()
    total_deaths = data_interface.load_total_deaths()
    total_deaths = total_deaths[total_deaths.location_id.isin(locations)].set_index('location_id')

    beta_scaling = forecast_spec.scenarios[scenario_name].beta_scaling

    def compute_beta_scaling_parameters(draw_id: int):
        # Construct a list of pandas Series indexed by location and named
        # as their column will be in the output dataframe. We'll append
        # to this list as we construct the parameters.
        draw_data = [total_deaths.copy(),
                     pd.Series(beta_scaling['window_size'], index=total_deaths.index, name='window_size')]

        # Today in the data is unique by draw.  It's a combination of the
        # number of predicted days from the elastispliner in the ODE fit
        # and the random draw of lag between infection and death from the
        # infectionator. Don't compute, let's look it up.
        dates_df = data_interface.load_dates_df(draw_id)
        transition_date = dates_df.set_index('location_id').sort_index()['end_date'].rename('date')

        beta_regression_df = data_interface.load_beta_regression(draw_id)
        beta_regression_df = beta_regression_df.set_index('location_id').sort_index()
        idx = beta_regression_df.index

        # Select out the transition day to compute the initial scaling parameter.
        beta_transition = beta_regression_df.loc[beta_regression_df['date'] == transition_date.loc[idx]]
        draw_data.append(beta_transition['beta'].rename('fit_final'))
        draw_data.append(beta_transition['beta_pred'].rename('pred_start'))
        draw_data.append((beta_transition['beta'] / beta_transition['beta_pred']).rename('scale_init'))

        # Compute the beta residual mean for our parameterization and hang on
        # to some ancillary information that may be useful for
        # plotting/debugging.
        rs = np.random.RandomState(draw_id)
        a = rs.randint(1, beta_scaling['average_over_min'])
        b = rs.randint(a + 7, beta_scaling['average_over_max'])

        draw_data.append(pd.Series(a, index=total_deaths.index, name='history_days_start'))
        draw_data.append(pd.Series(b, index=total_deaths.index, name='history_days_end'))

        beta_past = (beta_regression_df
                     .loc[beta_regression_df['date'] <= transition_date.loc[idx]]
                     .reset_index()
                     .set_index(['location_id', 'date'])
                     .sort_index())

        log_beta_resid_mean = (np.log(beta_past['beta'] / beta_past['beta_pred'])
                               .groupby(level='location_id')
                               .apply(lambda x: x.iloc[-b: -a].mean())
                               .rename('log_beta_residual_mean'))
        draw_data.append(log_beta_resid_mean)

        return pd.concat(draw_data, axis=1)

    draws = list(range(data_interface.get_n_draws()))
    with multiprocessing.Pool(FORECAST_SCALING_CORES) as pool:
        scaling_data = pool.imap(compute_beta_scaling_parameters, draws)

    import pdb; pdb.set_trace()



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
