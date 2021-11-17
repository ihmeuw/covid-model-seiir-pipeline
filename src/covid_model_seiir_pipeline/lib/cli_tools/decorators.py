from pathlib import Path
from typing import NamedTuple, Union

import click
import inflection


MaybePathLike = Union[str, Path, None]


class VersionInfo(NamedTuple):
    """Tiny struct for processing input versions from cli args and specs."""
    cli_arg: MaybePathLike
    spec_arg: MaybePathLike
    default: MaybePathLike
    metadata_key: str
    allow_default: bool

###########################
# Specification arguments #
###########################


def with_specification(specification_class):
    def _callback(ctx, param, value):
        return specification_class.from_path(value)
    return click.argument(
        'specification',
        type=click.Path(exists=True, dir_okay=False),
        callback=_callback
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


def with_version(default_root: Path, allow_default: bool = True, name: str = None):
    if name is None:
        name = inflection.underscore(default_root.name)

    def _callback(ctx, param, value):
        specification = ctx.params['specification']
        return VersionInfo(
            value,
            getattr(specification.data, f'{name}_version'),
            default_root,
            f'{name}_metadata',
            allow_default,
        )
    return click.option(
        f'--{inflection.dasherize(name)}-version',
        type=click.Path(exists=True, file_okay=False),
        help=f'Which version of {name} to use.'
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
