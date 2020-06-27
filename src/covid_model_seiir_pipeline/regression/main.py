from pathlib import Path
from shutil import copyfile

from covid_shared import cli_tools
from loguru import logger

from covid_model_seiir_pipeline import paths
from covid_model_seiir_pipeline.regression.specification import RegressionSpecification
from covid_model_seiir_pipeline.regression.data import RegressionDataInterface
from covid_model_seiir_pipeline.regression.workflow import RegressionWorkflow


def do_beta_regression(app_metadata: cli_tools.Metadata,
                       regression_specification: RegressionSpecification):
    logger.debug('Starting beta regression.')
    # init high level objects
    regression_paths = paths.RegressionPaths(Path(regression_specification.data.output_root), read_only=False)
    ode_paths = paths.RegressionPaths(Path(regression_specification.data.ode_fit_version))
    covariate_paths = paths.CovariatePaths(Path(regression_specification.data.covariate_version))

    data_interface = RegressionDataInterface(
        regression_paths=regression_paths,
        ode_paths=ode_paths,
        covariate_paths=covariate_paths
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
    regression_wf = RegressionWorkflow(regression_specification.data.output_root)
    regression_wf.attach_beta_regression_tasks()
    regression_wf.run()
