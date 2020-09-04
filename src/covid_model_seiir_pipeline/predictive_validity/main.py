from pathlib import Path

from covid_shared import cli_tools, shell_tools
from loguru import logger

from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.forecasting.specification import ForecastSpecification
from covid_model_seiir_pipeline.predictive_validity.specification import PredictiveValiditySpecification


def do_predictive_validity(app_metadata: cli_tools.Metadata,
                           regression_specification: RegressionSpecification,
                           forecast_specification: ForecastSpecification,
                           predictive_validity_specification: PredictiveValiditySpecification):
    logger.debug('Starting predictive validity.')

    output_root = Path(predictive_validity_specification.output_root)
    regression_root = output_root / 'regression'
    shell_tools.mkdir(regression_root)
    forecast_root = output_root / 'forecast'
    shell_tools.mkdir(forecast_root)

    scenario = predictive_validity_specification.forecast_scenario

    for holdout_version in predictive_validity_specification.holdout_versions:
        for alphas in predictive_validity_specification.alphas:
            regression_specification.data.infection_version = holdout_version.infectionator_version
            regression_specification.data.covariate_version = holdout_version.covariate_version
            regression_specification.parameters.alpha = alphas
            regression_dir_name = f'holdout_{holdout_version.holdout_days}_alpha_{alphas[0]}_{alphas[1]}'
            regression_dir = regression_root / regression_dir_name
            regression_specification.data.output_root = regression_dir
            shell_tools.mkdir(regression_dir)
            regression_spec_path = regression_dir / 'regression_specification.yaml'
            regression_specification.dump(regression_spec_path)

            for theta in predictive_validity_specification.thetas:
                for average_over_max in predictive_validity_specification.beta_scaling_average_over_maxes:
                    forecast_specification.scenarios[scenario].theta = theta
                    forecast_specification.scenarios[scenario].beta_scaling['average_over_max'] = average_over_max
                    forecast_dir_name = f'{regression_dir_name}_theta_{theta}_avg_over_max_{average_over_max}'
                    forecast_dir = forecast_root / forecast_dir_name
                    shell_tools.mkdir(forecast_dir)
                    forecast_spec_path = forecast_dir / 'forecast_specification.yaml'
                    forecast_specification.dump(forecast_spec_path)
