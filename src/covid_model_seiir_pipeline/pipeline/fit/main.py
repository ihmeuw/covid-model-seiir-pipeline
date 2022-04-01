from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)

from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.workflow import FitWorkflow
from covid_model_seiir_pipeline.pipeline.fit.model import postprocess, plotter


def do_fit(run_metadata: cli_tools.RunMetadata,
           specification: FitSpecification,
           output_root: Optional[str], mark_best: bool, production_tag: str,
           preprocess_only: bool,
           with_debugger: bool,
           input_versions: Dict[str, cli_tools.VersionInfo]) -> FitSpecification:
    specification, run_metadata = cli_tools.resolve_version_info(specification, run_metadata, input_versions)

    output_root = cli_tools.get_output_root(output_root, specification.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    specification.data.output_root = str(run_directory)

    run_metadata['output_path'] = str(run_directory)
    run_metadata['fit_specification'] = specification.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(fit_main,
                                         logger, with_debugger)
    app_metadata, _ = main(specification, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)
    return specification


def fit_main(app_metadata: cli_tools.Metadata,
             specification: FitSpecification,
             preprocess_only: bool):
    logger.info(f'Starting fit for version {specification.data.output_root}.')
    # init high level objects
    data_interface = FitDataInterface.from_specification(specification)

    # build directory structure and save metadata
    data_interface.make_dirs()
    data_interface.save_specification(specification)

    data_interface.save_summary(postprocess.get_data_dictionary(), 'data_dictionary')

    # build workflow and launch
    if not preprocess_only:
        workflow = FitWorkflow(specification.data.output_root, specification.workflow)
        plot_types = [plotter.PLOT_TYPE.model_fit, plotter.PLOT_TYPE.model_fit_tail]
        if specification.data.compare_version:
            plot_types.append(plotter.PLOT_TYPE.model_compare)
        workflow.attach_tasks(n_draws=data_interface.get_n_draws(),
                              n_oversample_draws=data_interface.get_n_oversample_draws(),
                              measures=list(postprocess.MEASURES),
                              plot_types=plot_types)
        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')

        name_map = data_interface.load_hierarchy('pred').set_index('location_id').location_name
        total_failures = data_interface.load_draw_resampling_map()['unrecoverable_pct']
        total_failures_formatted = '\n'.join([
            f'{name_map.loc[location_id]} ({location_id}): {failure_pct}' 
            for location_id, failure_pct in total_failures.items()
        ])
        if total_failures:
            logger.warning("The following locations failed in all measures in too "
                           f"many draws to resample:\n {total_failures_formatted}.")
            logger.warning(f"Flat list of failures: {list(total_failures)}")

    logger.info(f'Fit version {specification.data.output_root} complete.')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(FitSpecification)
@cli_tools.add_output_options(paths.SEIR_FIT_ROOT)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_version(paths.SEIR_PREPROCESS_ROOT)
@cli_tools.with_version(paths.SEIR_FIT_ROOT, allow_default=False, name='compare')
def fit(run_metadata: cli_tools.RunMetadata,
        specification: FitSpecification,
        output_root: str, mark_best: bool, production_tag: str,
        preprocess_only: bool,
        verbose: int, with_debugger: bool,
        **input_versions: cli_tools.VersionInfo):
    cli_tools.configure_logging_to_terminal(verbose)
    do_fit(
        run_metadata=run_metadata,
        specification=specification,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
        input_versions=input_versions,
    )

    logger.info('**Done**')
