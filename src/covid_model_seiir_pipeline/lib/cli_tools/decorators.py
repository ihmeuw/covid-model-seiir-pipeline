import click


with_regression_version = click.option(
    '--regression-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"regression_specification.yaml".'
)

with_forecast_version = click.option(
    '--forecast-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"forecast_specification.yaml".'
)

with_postprocessing_version = click.option(
    '--postprocessing-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"postprocessing_specification.yaml".'
)

with_diagnostics_version = click.option(
    '--diagnostics-version', '-i',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help='Full path to an existing directory containing a '
         '"diagnostics_specification.yaml".'
)

with_scenario = click.option(
    '--scenario', '-s',
    type=click.STRING,
    required=True,
    help='The scenario to be run.'
)

with_measure = click.option(
    '--measure', '-m',
    type=click.STRING,
    required=True,
    help='The measure to be run.'
)

with_draw_id = click.option(
    '--draw-id', '-d',
    type=click.INT,
    required=True,
    help='The draw to be run.'
)

with_name = click.option(
    '--name', '-n',
    type=click.INT,
    required=True,
    help='The name to be run.'
)
