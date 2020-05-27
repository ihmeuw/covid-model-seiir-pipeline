from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional


import numpy as np

from covid_model_seiir.ode_model import ODEProcessInput
from covid_model_seiir.model_runner import ModelRunner

from covid_model_seiir_pipeline.ode_fit import FitSpecification
from covid_model_seiir_pipeline.ode_fit.data import ODEDataInterface
from covid_model_seiir_pipeline.static_vars import INFECTION_COL_DICT


log = logging.getLogger(__name__)


def run_ode_fit(draw_id: int, ode_version: str):
    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    ode_fit_spec: FitSpecification = FitSpecification.from_path(
        Path(ode_version) / "fit_specification.yaml"
    )
    ode_data_interface = ODEDataInterface(ode_fit_spec.data)

    # Load data
    location_ids = ode_data_interface.load_location_ids()
    location_data = ode_data_interface.load_all_location_data(location_ids=location_ids,
                                                              draw_id=draw_id)

    # This seed is so that the alpha, sigma, gamma1 and gamma2 parameters are reproducible
    np.random.seed(draw_id)

    beta_fit_inputs = ODEProcessInput(
        df_dict=location_data,
        col_date=INFECTION_COL_DICT['COL_DATE'],
        col_cases=INFECTION_COL_DICT['COL_CASES'],
        col_pop=INFECTION_COL_DICT['COL_POP'],
        col_loc_id=INFECTION_COL_DICT['COL_LOC_ID'],
        col_lag_days=INFECTION_COL_DICT['COL_ID_LAG'],
        col_observed=INFECTION_COL_DICT['COL_OBS_DEATHS'],
        alpha=ode_fit_spec.parameters.alpha,
        sigma=ode_fit_spec.parameters.sigma,
        gamma1=ode_fit_spec.parameters.gamma1,
        gamma2=ode_fit_spec.parameters.gamma2,
        solver_dt=ode_fit_spec.parameters.solver_dt,
        spline_options={
            'spline_knots': ode_fit_spec.parameters.knots,
            'spline_degree': ode_fit_spec.parameters.degree,
            'prior_spline_convexity': ode_fit_spec.parameters.concavity,
            'prior_spline_monotonicity': ode_fit_spec.parameters.increasing,
        },
        day_shift=ode_fit_spec.parameters.day_shift,
        spline_se_power=ode_fit_spec.parameters.spline_se_power,
        spline_space=ode_fit_spec.parameters.spline_space,
    )

    # ----------------------- BETA SPLINE + ODE -------------------------------- #
    # Start a Model Runner with the processed inputs and fit the beta spline / ODE
    mr = ModelRunner()
    mr.fit_beta_ode(beta_fit_inputs)

    # Save location-specific beta fit (compartment) files for easy reading later
    beta_fit = mr.get_beta_ode_fit()
    for l_id in location_ids:
        loc_beta_fits = beta_fit.loc[beta_fit[INFECTION_COL_DICT['COL_LOC_ID']] == l_id].copy()
        ode_data_interface.save_draw_beta_fit_file(loc_beta_fits, l_id, draw_id)

    # Save the parameters of alpha, sigma, gamma1, and gamma2 that were drawn
    draw_beta_params = mr.get_beta_ode_params()
    beta_start_end_dates = mr.get_beta_start_end_dates()
    ode_data_interface.save_draw_beta_param_file(draw_beta_params, draw_id)
    ode_data_interface.save_draw_date_file(beta_start_end_dates, draw_id)


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
