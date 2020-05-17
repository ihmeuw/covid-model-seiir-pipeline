from pathlib import Path

import click
from covid_shared import cli_tools, paths
from loguru import logger

from seiir_model_pipeline import regression


@click.group()
def seiir():
    pass


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('regression_specification',
                type=click.Path(exists=True, dir_okay=False))
@click.option('--infection-version',
              type=click.Path(file_okay=False),
              help="Which version of infectionator inputs to use in the"
                   "regression.")
@click.option('--covariates-version',
              type=click.Path(file_okay=False),
              help=('Which version of the covariates to use in the '
                    'regression.'))
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              show_default=True)
@click.option('-b', '--mark-best', 'mark_dir_as_best',
              is_flag=True,
              help='Marks the new outputs as best in addition to marking '
                   'them as latest.')
@click.option('-p', '--production-tag',
              type=click.STRING,
              help='Tags this run as a production run.')
@cli_tools.add_verbose_and_with_debugger
def regress(run_metadata,
            regression_specification,
            infection_version, covariates_version,
            output_root, mark_dir_as_best, production_tag,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    regression_spec = regression.load_regression_specification(regression_specification)

    if infection_version:
        pass
    elif regression_spec.data.infection_version:
        infection_version = regression_spec.data.infection_version
    else:
        infection_version = paths.BEST_LINK

    infection_root = cli_tools.get_last_stage_directory(
        infection_version, last_stage_root=paths.INFECTIONATOR_OUTPUTS
    )
    regression_spec.data.infection_version = str(infection_root)

    if covariates_version:
        pass
    elif regression_spec.data.covariate_version:
        covariates_version = regression_spec.data.covariate_version
    else:
        covariates_version = paths.BEST_LINK

    covariates_root = cli_tools.get_last_stage_directory(
        covariates_version, last_stage_root=paths.SEIR_COVARIATES_OUTPUT_ROOT
    )
    regression_spec.data.covariate_version = str(covariates_root)

    for key, input_root in zip(['infection_inputs', 'covariates_inputs'],
                               [infection_root, covariates_root]):
        run_metadata.update_from_path(key, input_root / paths.METADATA_FILE_NAME)

    if output_root:
        output_root = Path(output_root).resolve()
    elif regression_spec.data.output_root:
        output_root = Path(regression_spec.data.output_root).resolve()
    else:
        output_root = paths.SEIR_REGRESSION_OUTPUTS
    regression_spec.data.output_root = str(output_root)

    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)
    run_metadata['regression_specification'] = regression_spec.to_dict()

    main = cli_tools.monitor_application(regression.do_beta_regression,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, infection_root,
                           covariates_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')
