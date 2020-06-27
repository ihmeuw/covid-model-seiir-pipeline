from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional

import numpy as np

from covid_model_seiir_pipeline.ode_fit import model
from covid_model_seiir_pipeline.ode_fit.data import ODEDataInterface
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT


log = logging.getLogger(__name__)


def run_ode_fit(draw_id: int, ode_version: str):
    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    fit_specification: FitSpecification = FitSpecification.from_path(
        Path(ode_version) / "fit_specification.yaml"
    )
    data_interface = ODEDataInterface.from_specification(fit_specification)

    # Load data
    location_ids = data_interface.load_location_ids()
    location_data = data_interface.load_all_location_data(location_ids=location_ids,
                                                          draw_id=draw_id)

    # This seed is so that the alpha, sigma, gamma1 and gamma2 parameters are reproducible
    np.random.seed(draw_id)

    beta_fit_inputs = model.ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
        col_observed=INFECTION_COL_DICT['COL_OBS_DEATHS'],
        alpha=fit_specification.parameters.alpha,
        sigma=fit_specification.parameters.sigma,
        gamma1=fit_specification.parameters.gamma1,
        gamma2=fit_specification.parameters.gamma2,
        solver_dt=fit_specification.parameters.solver_dt,
        day_shift=fit_specification.parameters.day_shift,

    )

    # Convert infections into beta and fit compartments to data.
    ode_model = model.ODEProcess(beta_fit_inputs)
    ode_model.process()

    # Save location-specific beta fit (compartment) files for easy reading later
    beta_fit = ode_model.create_result_df()
    for l_id in location_ids:
        loc_beta_fits = beta_fit.loc[beta_fit[INFECTION_COL_DICT['COL_LOC_ID']] == l_id].copy()
        data_interface.save_draw_beta_fit_file(loc_beta_fits, l_id, draw_id)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    draw_beta_params = ode_model.create_params_df()
    beta_start_end_dates = ode_model.create_start_end_date_df()
    data_interface.save_draw_beta_param_file(draw_beta_params, draw_id)
    data_interface.save_draw_date_file(beta_start_end_dates, draw_id)


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


def main():
    args = parse_arguments()
    run_ode_fit(draw_id=args.draw_id, ode_version=args.ode_version)


if __name__ == '__main__':
    main()
