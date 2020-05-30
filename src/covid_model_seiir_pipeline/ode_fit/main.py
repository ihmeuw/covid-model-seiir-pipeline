"""Runner for the beta ODE fit."""
from pathlib import Path

from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline.ode_fit import FitSpecification
from covid_model_seiir_pipeline.paths import ODEPaths
from covid_model_seiir_pipeline.ode_fit.data import ODEDataInterface
from covid_model_seiir_pipeline.ode_fit.workflow import ODEFitWorkflow


def do_beta_fit(app_metadata: cli_tools.Metadata,
                fit_specification: FitSpecification,
                output_root: Path):
    logger.debug('Starting Beta fit.')
    # init high level objects
    ode_paths = ODEPaths(output_root, read_only=False)
    data_interface = ODEDataInterface(
        ode_fit_root=Path(fit_specification.data.output_root),
        infection_root=Path(fit_specification.data.infection_version),
        location_file=(
            Path('/ihme/covid-19/seir-pipeline-outputs/metadata-inputs') /
            f'location_metadata_{fit_specification.data.location_set_version_id}.csv'
        )
    )

    # build directory structure
    location_ids = data_interface.load_location_ids()
    ode_paths.make_dirs(location_ids)

    # build workflow and launch
    ode_wf = ODEFitWorkflow(fit_specification)
    ode_wf.attach_ode_tasks()
    ode_wf.run()
