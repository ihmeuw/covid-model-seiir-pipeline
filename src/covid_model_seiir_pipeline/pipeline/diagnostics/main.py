from typing import Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger
import yaml

from covid_model_seiir_pipeline.lib import cli_tools
from covid_model_seiir_pipeline.lib import io
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import DiagnosticsSpecification
from covid_model_seiir_pipeline.pipeline.diagnostics.workflow import DiagnosticsWorkflow


def do_diagnostics(run_metadata: cli_tools.RunMetadata,
                   diagnostics_specification: str,
                   output_root: Optional[str], mark_best: bool, production_tag: str,
                   preprocess_only: bool,
                   with_debugger: bool) -> DiagnosticsSpecification:
    diagnostics_spec = DiagnosticsSpecification.from_path(diagnostics_specification)

    outputs_versions = set()
    for grid_plot_spec in diagnostics_spec.grid_plots:
        for comparator in grid_plot_spec.comparators:
            comparator_version_path = cli_tools.get_input_root(None,
                                                               comparator.version,
                                                               paths.SEIR_FINAL_OUTPUTS)
            outputs_versions.add(comparator_version_path)
            comparator.version = str(comparator_version_path)

    if diagnostics_spec.cumulative_deaths_compare_csv:
        for comparator in diagnostics_spec.cumulative_deaths_compare_csv.comparators:
            comparator_version_path = cli_tools.get_input_root(None,
                                                               comparator.version,
                                                               paths.SEIR_FINAL_OUTPUTS)
            outputs_versions.add(comparator_version_path)
            comparator.version = str(comparator_version_path)
    for scatters_spec in diagnostics_spec.scatters:
        x_axis_version_path = cli_tools.get_input_root(None,
                                                       scatters_spec.x_axis.version,
                                                       paths.SEIR_FINAL_OUTPUTS)
        scatters_spec.x_axis.version = str(x_axis_version_path)
        outputs_versions.add(x_axis_version_path)
        y_axis_version_path = cli_tools.get_input_root(None,
                                                       scatters_spec.y_axis.version,
                                                       paths.SEIR_FINAL_OUTPUTS)
        scatters_spec.y_axis.version = str(y_axis_version_path)
        outputs_versions.add(y_axis_version_path)

    output_root = cli_tools.get_output_root(output_root,
                                            diagnostics_spec.data.output_root)
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)

    diagnostics_spec.data.output_root = str(run_directory)

    outputs_metadata = []
    for output_version in outputs_versions:
        with (output_version / paths.METADATA_FILE_NAME).open() as metadata_file:
            outputs_metadata.append(yaml.full_load(metadata_file))

    run_metadata['seir_outputs_metadata'] = outputs_metadata
    run_metadata['output_path'] = str(run_directory)
    run_metadata['diagnostics_specification'] = diagnostics_spec.to_dict()

    cli_tools.configure_logging_to_files(run_directory)
    # noinspection PyTypeChecker
    main = cli_tools.monitor_application(diagnostics_main,
                                         logger, with_debugger)
    app_metadata, _ = main(diagnostics_spec, preprocess_only)

    cli_tools.finish_application(run_metadata, app_metadata,
                                 run_directory, mark_best, production_tag)
    return diagnostics_spec


def diagnostics_main(app_metadata: cli_tools.Metadata,
                     diagnostics_specification: DiagnosticsSpecification,
                     preprocess_only: bool):
    logger.info(f'Starting diagnostics for version {diagnostics_specification.data.output_root}.')

    diagnostics_root = io.DiagnosticsRoot(diagnostics_specification.data.output_root)
    io.dump(diagnostics_specification.to_dict(), diagnostics_root.specification())

    if not preprocess_only:
        workflow = DiagnosticsWorkflow(diagnostics_specification.data.output_root,
                                       diagnostics_specification.workflow)
        grid_plot_jobs = [grid_plot_spec.name for grid_plot_spec in diagnostics_specification.grid_plots]
        compare_csv = bool(diagnostics_specification.cumulative_deaths_compare_csv)
        scatters_jobs = [scatters_spec.name for scatters_spec in diagnostics_specification.scatters]

        workflow.attach_tasks(grid_plot_jobs, compare_csv, scatters_jobs)

        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete')


@click.command()
@cli_tools.pass_run_metadata()
@cli_tools.with_specification(DiagnosticsSpecification)
@cli_tools.add_output_options(paths.SEIR_DIAGNOSTICS_OUTPUTS)
@cli_tools.add_preprocess_only
@cli_tools.add_verbose_and_with_debugger
def diagnostics(run_metadata,
                diagnostics_specification,
                output_root, mark_best, production_tag,
                preprocess_only,
                verbose, with_debugger):
    cli_tools.configure_logging_to_terminal(verbose)

    do_diagnostics(
        run_metadata=run_metadata,
        diagnostics_specification=diagnostics_specification,
        output_root=output_root,
        mark_best=mark_best,
        production_tag=production_tag,
        preprocess_only=preprocess_only,
        with_debugger=with_debugger,
    )

    logger.info('**Done**')
