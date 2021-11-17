import click

###########################
# Specification arguments #
###########################


def with_specification(specification_class):
    def _callback(ctx, param, value):
        return specification_class.from_path(value)
    return click.argument(
        'specification',
        type=click.Path(exists=True, dir_okay=False),
    )


###########################
# Main input data options #
###########################

with_location_specification = click.option(
    '-l', '--location-specification',
    type=click.STRING,
    help='Either a location set version id used to pull a list of '
         'locations to run, or a full path to a file describing '
         'the location set.',
)
with_model_inputs_version = click.option(
    '--model-inputs-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of model-inputs to use.',
)
with_age_specific_rates_version = click.option(
    '--age-specific-rates-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of age-specific-rates to use.',
)
with_mortality_scalars_version = click.option(
    '--mortality-scalars-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of mortality-scalars to use.',
)
with_mask_use_version = click.option(
    '--mask-use-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of mask use to use.',
)
with_mobility_version = click.option(
    '--mobility-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of mobility to use.',
)
with_pneumonia_version = click.option(
    '--pneumonia-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of pneumonia to use.',
)
with_population_density_version = click.option(
    '--population-density-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of population density to use.',
)
with_testing_version = click.option(
    '--testing-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of testing to use.',
)
with_variant_prevalence_version = click.option(
    '--variant-prevalence-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of variant prevalence to use.',
)
with_vaccine_coverage_version = click.option(
    '--vaccine-coverage-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of vaccine coverage to use.',
)
with_vaccine_efficacy_version = click.option(
    '--vaccine-efficacy-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of vaccine efficacy to use.',
)

with_infection_version = click.option(
    '--infection-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of infection inputs to use.',
)
with_covariates_version = click.option(
    '--covariates-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of covariates to use.',
)
with_waning_version = click.option(
    '--preprocessing-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of preprocessing to use.',
)
with_variant_version = click.option(
    '--variant-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of variants to use.',
)
with_mortality_ratio_version = click.option(
    '--mortality-ratio-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of the mortality age pattern to use.',
)
with_priors_version = click.option(
    '--priors-version',
    type=click.Path(exists=True, file_okay=False),
    help='Data based priors for the regression.',
)
with_coefficient_version = click.option(
    '--coefficient-version',
    type=click.Path(exists=True, file_okay=False),
    help='A prior regression version for pinning the regression '
         'coefficients. If provided, all fixed effects from the '
         'provided version will be used and only random effects will '
         'be fit.',
)

with_regression_version = click.option(
    '--regression-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of beta regression to use.',
)
with_forecast_version = click.option(
    '--forecast-version',
    type=click.Path(exists=True, file_okay=False),
    help='Which version of forecasts to use.'
)

######################
# Other main options #
######################

add_preprocess_only = click.option(
    '--preprocess-only',
    is_flag=True,
    help='Only make the directory and set up the metadata. '
         'Useful for setting up output directories for testing '
         'tasks individually.',
)

########################
# Task version options #
########################

with_task_preprocessing_version = click.option(
    '--preprocessing-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"preprocessing_specification.yaml".',
)
with_task_fit_version = click.option(
    '--fit-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"fit_specification.yaml".',
)
with_task_regression_version = click.option(
    '--regression-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"regression_specification.yaml".',
)
with_task_forecast_version = click.option(
    '--forecast-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"forecast_specification.yaml".',
)
with_task_postprocessing_version = click.option(
    '--postprocessing-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"postprocessing_specification.yaml".',
)
with_task_diagnostics_version = click.option(
    '--diagnostics-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"diagnostics_specification.yaml".',
)

#########################
# Task argument options #
#########################

with_scenario = click.option(
    '--scenario', '-s',
    type=click.STRING,
    required=True,
    help='The scenario to be run.',
)
with_measure = click.option(
    '--measure', '-m',
    type=click.STRING,
    required=True,
    help='The measure to be run.',
)
with_draw_id = click.option(
    '--draw-id', '-d',
    type=click.INT,
    required=True,
    help='The draw to be run.',
)
with_name = click.option(
    '--name', '-n',
    type=click.STRING,
    required=True,
    help='The name to be run.',
)
with_progress_bar = click.option(
    '--pb', 'progress_bar',
    is_flag=True,
    help='Whether to show progress bars.',
)
