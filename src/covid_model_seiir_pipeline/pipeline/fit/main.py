import functools
from typing import Dict, Optional

import click
from covid_shared import ihme_deps, paths
from loguru import logger
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
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
                              measures=list(postprocess.MEASURES),
                              plot_types=plot_types)
        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')

        # Check for bad locations
        total_population = data_interface.load_population(measure='total').population

        _runner = functools.partial(
            get_broken_locations,
            data_interface=data_interface,
            total_population=total_population,
        )
        results = parallel.run_parallel(
            _runner,
            arg_list=list(range(data_interface.get_n_draws())),
            num_cores=26,
            progress_bar=True
        )
        report = make_broken_location_report(results)
        if report:
            logger.warning(report)

    logger.info(f'Fit version {specification.data.output_root} complete.')


def get_broken_locations(draw_id: int,
                         data_interface: FitDataInterface,
                         total_population: pd.Series):
    infections = data_interface.load_posterior_epi_measures(
        draw_id, columns=['daily_naive_unvaccinated_infections', 'round']
    )
    infections = infections.loc[infections['round'] == 2, 'daily_naive_unvaccinated_infections']

    below_0 = (infections.groupby('location_id').min() < 0).rename(f'draw_{draw_id}')

    over_total_pop = infections.groupby('location_id').sum()
    over_total_pop = ((over_total_pop / total_population.reindex(over_total_pop.index)) > 1).rename(f'draw_{draw_id}')

    return below_0, over_total_pop


def make_broken_location_report(broken_locations):
    below_0, over_total_pop = zip(*broken_locations)
    below_0 = pd.concat(below_0, axis=1)
    draws = below_0.columns
    below_0 = below_0.reset_index()
    over_total_pop = pd.concat(over_total_pop, axis=1).reset_index()
    
    report = ''
    overall_missing = set()
    overall_below_0 = set()
    overall_over_total_pop = set()
    for draw in draws:
        draw_report = ''
        
        missing = below_0.loc[below_0[draw].isnull(), 'location_id'].tolist()
        draw_below_0 = below_0.loc[below_0[draw].fillna(False), 'location_id'].tolist()
        draw_over_total_pop = over_total_pop.loc[over_total_pop[draw].fillna(False), 'location_id'].tolist()
        if missing:
            draw_report += f'    missing: {missing}\n'
            overall_missing |= set(missing)
        if draw_below_0:
            draw_report += f'    below_0: {draw_below_0}\n'
            overall_below_0 |= set(draw_below_0)
        if draw_over_total_pop:
            draw_report += f'    over_total_pop: {draw_over_total_pop}\n'
            overall_over_total_pop |= set(draw_over_total_pop)
        if draw_report:
            report += f'{draw}:\n' + draw_report
    if report:
        report = 'Failing locations found for some draws\n' + report
        report += 'Overall:\n'
        if overall_missing:
            report += f'    missing: {list(overall_missing)}\n'
        if overall_below_0:
            report += f'    below_0: {list(overall_below_0)}\n'
        if overall_over_total_pop:
            report += f'    over_total_pop: {list(overall_over_total_pop)}\n'
        report += f'    all_problem_locs: {list(overall_missing | overall_below_0 | overall_over_total_pop)}\n'
    return report


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
