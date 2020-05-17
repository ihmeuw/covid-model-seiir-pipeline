from pathlib import Path

import click
from covid_shared import cli_tools, paths
from loguru import logger

from seiir_model_pipeline import regress


@click.group()
def seiir():
    pass


@seiir.command()
@cli_tools.pass_run_metadata()
@click.argument('regression_specification',
                type=click.Path(exists=True, dir_okay=False),
                help='Specification of regression parameters.')
@click.option('--infectionator-version',
              type=click.Choice([paths.BEST_LINK, paths.LATEST_LINK]),
              default=paths.BEST_LINK,
              help="Which version of infectionator inputs to use in the"
                   "regression. Ignored if '--infectionator-root' is "
                   "provided.")
@click.option('--infectionator-root',
              type=click.Path(file_okay=False),
              help='Specific directory containing infectionator'
                   'outputs to use in the regression.')
@click.option('--covariates-version',
              type=click.Choice([paths.BEST_LINK, paths.LATEST_LINK]),
              default=paths.BEST_LINK,
              help=('Which version of the covariates to use in the '
                    'regression. Ignored if "--covariates-root" is provided.'))
@click.option('--covariates-root',
              type=click.Path(file_okay=False),
              help='Specific directory containing the covariates data'
                   'to use in the regression.')
@click.option('-o', '--output-root',
              type=click.Path(file_okay=False),
              default=paths.SEIR_REGRESSION_OUTPUTS,
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
            infectionator_version, infectionator_root,
            covariates_version, covariates_root,
            output_root, mark_dir_as_best, production_tag,
            verbose, with_debugger):
    """Perform beta regression for a set of infections and covariates."""
    cli_tools.configure_logging_to_terminal(verbose)

    infectionator_root = cli_tools.get_last_stage_directory(
        infectionator_version, infectionator_root, paths.INFECTIONATOR_OUTPUTS
    )
    covariates_root = cli_tools.get_last_stage_directory(
        covariates_version, covariates_root, paths.SEIR_COVARIATES_OUTPUT_ROOT
    )
    for key, input_root in zip(['infectionator_inputs', 'covariates_inputs'],
                               [infectionator_root, covariates_root]):
        run_metadata.update_from_path(key, input_root / paths.METADATA_FILE_NAME)

    output_root = Path(output_root).resolve()
    cli_tools.setup_directory_structure(output_root, with_production=True)
    run_directory = cli_tools.make_run_directory(output_root)
    run_metadata['output_path'] = str(run_directory)
    cli_tools.configure_logging_to_files(run_directory)

    regression_spec = regress.load_regression_specification(regression_specification)
    run_metadata['regression_specification'] = regression_spec.to_dict()

    main = cli_tools.monitor_application(regress.do_beta_regression,
                                         logger, with_debugger)
    app_metadata, _ = main(regression_spec, infectionator_root,
                           covariates_root, run_directory)

    run_metadata['app_metadata'] = app_metadata.to_dict()
    run_metadata.dump(run_directory / 'metadata.yaml')

    cli_tools.make_links(app_metadata, run_directory, mark_dir_as_best, production_tag)

    logger.info('**Done**')
