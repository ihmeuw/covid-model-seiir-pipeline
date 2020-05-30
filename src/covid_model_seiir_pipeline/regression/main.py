from pathlib import Path
from shutil import copyfile

from covid_shared import cli_tools

from covid_model_seiir_pipeline import static_vars
from covid_model_seiir_pipeline.paths import RegressionPaths
from covid_model_seiir_pipeline.ode_fit.specification import FitSpecification
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification,
                       output_dir: Path):
    ode_fit_spec: FitSpecification = FitSpecification.from_path(
        Path(regression_specification.data.ode_fit_version) / static_vars.FIT_SPECIFICATION_FILE
    )
    # init high level objects
    regression_paths = RegressionPaths(output_dir, read_only=False)
    data_interface = RegressionDataInterface(
        regression_root=Path(regression_specification.data.output_root),
        covariate_root=Path(regression_specification.data.covariate_version),
        ode_fit_root=Path(ode_fit_spec.data.output_root),
        infection_root=Path(ode_fit_spec.data.infection_version),
        location_file=(Path('/ihme/covid-19/seir-pipeline-outputs/metadata-inputs') /
                       f'location_metadata_{ode_fit_spec.data.location_set_version_id}.csv')
    )

    # build directory structure
    location_ids = data_interface.load_location_ids()
    regression_paths.make_dirs(location_ids)

    # copy info files
    for covariate in regression_specification.covariates.keys():
        info_files = data_interface.covariate_paths.get_info_files(covariate)
        for file in info_files:
            dest_path = regression_paths.info_dir / file.name
            copyfile(str(file), str(dest_path))

    # build workflow and launch
    regression_wf = RegressionWorkflow(regression_specification, ode_fit_spec)
    regression_wf.attach_regression_tasks()
    regression_wf.run()
