from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import shlex
from typing import Optional, List


import numpy as np
import pandas as pd

from seiir_model.ode_model import ODEProcessInput
from seiir_model.model_runner import ModelRunner

from seiir_model_pipeline.paths import ODEPaths, InfectionPaths
from seiir_model_pipeline.ode_fit.specification import FitSpecification
from seiir_model_pipeline.globals import INFECTION_COL_DICT


log = logging.getLogger(__name__)


class ODEDataInterface:

    def __init__(self, ode_fit_data: FitSpecification.data):
        self.ode_paths = ODEPaths(Path(ode_fit_data.output_root))
        self.infection_paths = InfectionPaths(Path(ode_fit_data.infection_version))

        # TODO: figure out where this comes from
        self.location_metadata_file = Path(
            '/ihme/covid-19/seir-pipeline-outputs/metadata-inputs/location_metadata_999.csv'
        )

    def load_location_ids(self) -> List[int]:
        return pd.read_csv(self.location_metadata_file)["location_id"].tolist()

    def load_all_location_data(self, location_ids: List[int], draw_id: int):
        dfs = dict()
        for loc in location_ids:
            file = self.get_infection_file(location_id=loc, draw_id=draw_id)
            dfs[loc] = pd.read_csv(file)

        # validate
        locs_na = []
        locs_neg = []
        for loc, df in dfs.items():
            if df[INFECTION_COL_DICT['COL_CASES']].isna().any():
                locs_na.append(loc)
            if (df[INFECTION_COL_DICT['COL_CASES']].to_numpy() < 0.0).any():
                locs_neg.append(loc)
        if len(locs_na) > 0 and len(locs_neg) > 0:
            raise ValueError(
                'NaN in infection data: ' + str(locs_na) + '. Negatives in infection data: ' +
                str(locs_neg)
            )
        if len(locs_na) > 0:
            raise ValueError('NaN in infection data: ' + str(locs_na))
        if len(locs_neg) > 0:
            raise ValueError('Negatives in infection data:' + str(locs_neg))

        return dfs

    def save_draw_beta_fit_file(self, df, location_id: int, draw_id: int):
        df.to_csv(self.ode_paths.get_draw_beta_fit_file(location_id, draw_id), index=False)

    def save_draw_beta_param_file(self, df, draw_id: int):
        df.to_csv(self.ode_paths.get_draw_beta_param_file(draw_id))


def run_ode_fit(draw_id: int, ode_version: str):
    # -------------------------- LOAD INPUTS -------------------- #
    # Load metadata
    ode_fit_spec = FitSpecification.from_path(Path(ode_version) / "specification.yaml")
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
        alpha=ode_fit_spec.parameters.alpha,
        sigma=ode_fit_spec.parameters.sigma,
        gamma1=ode_fit_spec.parameters.gamma1,
        gamma2=ode_fit_spec.parameters.gamma2,
        solver_dt=ode_fit_spec.parameters.solver_dt,
        spline_options={
            'spline_knots': ode_fit_spec.parameters.knots,
            'spline_degree': ode_fit_spec.parameters.degree
        },
        day_shift=ode_fit_spec.parameters.day_shift
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
    ode_data_interface.save_draw_beta_param_file(draw_beta_params, draw_id)


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
