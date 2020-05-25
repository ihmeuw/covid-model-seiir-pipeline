from argparse import ArgumentParser, Namespace
from typing import Optional
import shlex
import logging
import numpy as np

from covid_model_seiir.model_runner import ModelRunner

from covid_model_seiir_pipeline.core.versioner import INFECTION_COL_DICT
from covid_model_seiir_pipeline.core.versioner import load_ode_settings
from covid_model_seiir_pipeline.core.versioner import Directories

from covid_model_seiir_pipeline.core.data import load_all_location_data
from covid_model_seiir_pipeline.core.utils import load_locations
from covid_model_seiir_pipeline.core.model_inputs import process_ode_process_input

log = logging.getLogger(__name__)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--draw-id", type=int, required=True)
    parser.add_argument("--ode-version", type=str, required=True)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()

    return args


def run_ode_fit(draw_id: int, ode_version: str):

    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    directories = Directories(
        ode_version=ode_version,
        regression_version=None,
        forecast_version=None
    )
    settings = load_ode_settings(ode_version)

    # Load data
    location_ids = load_locations(directories)
    location_data = load_all_location_data(
        directories=directories, location_ids=location_ids, draw_id=draw_id
    )

    # This seed is so that the alpha, sigma, gamma1 and gamma2 parameters are reproducible
    np.random.seed(draw_id)
    beta_fit_inputs = process_ode_process_input(
        settings=settings,
        location_data=location_data,
    )

    # ----------------------- BETA SPLINE + ODE -------------------------------- #
    # Start a Model Runner with the processed inputs and fit the beta spline / ODE
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)

    beta_fit = mr.get_beta_ode_fit()

    # Save location-specific beta fit (compartment) files for easy reading later
    for l_id in location_ids:
        loc_beta_fits = beta_fit.loc[beta_fit[INFECTION_COL_DICT['COL_LOC_ID']] == l_id].copy()
        loc_beta_fits.to_csv(directories.get_draw_beta_fit_file(l_id, draw_id), index=False)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    mr.get_beta_ode_params().to_csv(directories.get_draw_beta_param_file(draw_id), index=False)

    # Save the start and end dates of the fit data
    mr.get_beta_start_end_dates().to_csv(directories.get_draw_dates_file(draw_id), index=False)


def main():

    args = parse_arguments()
    run_ode_fit(draw_id=args.draw_id, ode_version=args.ode_version)


if __name__ == '__main__':
    main()
