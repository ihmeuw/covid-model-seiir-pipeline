from argparse import ArgumentParser, Namespace
from typing import Optional, Dict
import logging
from pathlib import Path
import shlex

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.model_runner import ModelRunner
from covid_model_seiir_pipeline.forecasting.model import SeiirModelSpecs

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.covariate_model import convert_to_covmodel
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.forecasting.data import ForecastDataInterface
from covid_model_seiir_pipeline.forecasting.model import get_ode_init_cond


log = logging.getLogger(__name__)


def run_beta_forecast(location_id: int, forecast_version: str, scenario_name: str):
    # -------------------------- CONSTRUCT DATA INTERFACE AND SPECS -------------------- #
    log.info("Initiating SEIIR beta forecasting.")
    forecast_spec: ForecastSpecification = ForecastSpecification.from_path(
        Path(forecast_version) / static_vars.FORECAST_SPECIFICATION_FILE
    )
    scenario_spec = forecast_spec.scenarios[scenario_name]
    regress_spec: RegressionSpecification = RegressionSpecification.from_path(
        Path(forecast_spec.data.regression_version) / static_vars.REGRESSION_SPECIFICATION_FILE
    )
    ode_fit_spec: FitSpecification = FitSpecification.from_path(
        Path(regress_spec.data.ode_fit_version) / static_vars.FIT_SPECIFICATION_FILE
    )

    data_interface = ForecastDataInterface(
        forecast_root=Path(forecast_spec.data.output_root) / scenario_name,
        regression_root=Path(regress_spec.data.output_root),
        ode_fit_root=Path(ode_fit_spec.data.output_root),
        infection_root=Path(ode_fit_spec.data.infection_version),
        location_file=Path(ode_fit_spec.data.location_set_file)
    )

    # -------------------------- FORECAST THE BETA FORWARDS -------------------- #
    mr = ModelRunner()

    # Get all inputs for the beta forecasting
    # Get all inputs for the ODE
    scales = []

    for draw_id in range(ode_fit_spec.parameters.n_draws):
        print(f"On draw {draw_id}\n")

        # Load the previous beta fit compartments and ODE parameters
        beta_fit = data_interface.load_beta_fit(draw_id=draw_id, location_id=location_id)
        beta_params = data_interface.load_beta_params(draw_id=draw_id)

        # covariate pool standardizes names to be {covariate}_{scenario}
        scenario_covariate_mapping: Dict[str, str] = {}
        for covariate in regress_spec.covariates.values():
            if covariate.name != "intercept":
                scenario = scenario_spec.covariates[covariate.name]
                scenario_covariate_mapping[f"{covariate.name}_{scenario}"] = covariate.name

        # load covariates data
        covariate_df = data_interface.load_covariate_scenarios(
            draw_id=draw_id,
            location_id=location_id,
            scenario_covariate_mapping=scenario_covariate_mapping
        )

        # Convert settings to the covariates model
        _, all_covmodels_set = convert_to_covmodel(list(regress_spec.covariates.values()))

        # Figure out what date we need to forecast from (the end of the component fit in ode)
        beta_fit_date = pd.to_datetime(beta_fit[static_vars.INFECTION_COL_DICT['COL_DATE']])
        CURRENT_DATE = (beta_fit[beta_fit_date == beta_fit_date.max()]
                        )[static_vars.INFECTION_COL_DICT['COL_DATE']].iloc[0]
        cov_date = pd.to_datetime(covariate_df[static_vars.COVARIATE_COL_DICT['COL_DATE']])
        covariate_data = covariate_df.loc[cov_date >= beta_fit_date.max()].copy()

        # Load the regression coefficients
        regression_fit = data_interface.load_regression_coefficients(draw_id=draw_id)
        # Forecast the beta forward with those coefficients
        forecasts = mr.predict_beta_forward_prod(
            covmodel_set=all_covmodels_set,
            df_cov=covariate_data,
            df_cov_coef=regression_fit,
            col_t=static_vars.COVARIATE_COL_DICT['COL_DATE'],
            col_group=static_vars.COVARIATE_COL_DICT['COL_LOC_ID']
        )

        betas = forecasts.beta_pred.values
        days = forecasts[static_vars.COVARIATE_COL_DICT['COL_DATE']].values
        days = pd.to_datetime(days)
        times = np.array((days - days.min()).days)

        # Anchor the betas at the last observed beta (fitted)
        # and scale everything into the future from this anchor value
        anchor_beta = beta_fit.beta[beta_fit.date == CURRENT_DATE].iloc[0]
        scale = anchor_beta / betas[0]
        scales.append(scale)
        # scale = scale + (1 - scale)/20.0*np.arange(betas.size)
        # scale[21:] = 1.0
        betas = betas * scale

        # Get initial conditions based on the beta fit for forecasting into the future
        init_cond = get_ode_init_cond(
            beta_ode_fit=beta_fit,
            current_date=CURRENT_DATE,
            location_id=location_id
        ).astype(float)
        N = np.sum(init_cond)  # total population
        model_specs = SeiirModelSpecs(
            alpha=beta_params['alpha'],
            sigma=beta_params['sigma'],
            gamma1=beta_params['gamma1'],
            gamma2=beta_params['gamma2'],
            N=N
        )
        # Forecast all of the components based on the forecasted beta
        forecasted_components = mr.forecast(
            model_specs=model_specs,
            init_cond=init_cond,
            times=times,
            betas=betas,
            dt=ode_fit_spec.parameters.solver_dt
        )
        forecasted_components[static_vars.COVARIATE_COL_DICT['COL_DATE']] = days

        data_interface.save_components(
            df=forecasted_components,
            location_id=location_id,
            draw_id=draw_id
        )

    data_interface.save_beta_scales(scales, location_id)


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or a command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--location-id", type=int, required=True)
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
    run_beta_forecast(location_id=args.location_id,
                      forecast_version=args.forecast_version,
                      scenario_name=args.scenario_name)


if __name__ == '__main__':
    main()
