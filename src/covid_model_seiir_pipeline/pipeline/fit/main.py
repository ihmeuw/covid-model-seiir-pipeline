import functools
from typing import Dict, List, Optional

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
                              n_oversample_draws=data_interface.get_n_oversample_draws(),
                              measures=list(postprocess.MEASURES),
                              plot_types=plot_types)
        try:
            workflow.run()
        except ihme_deps.WorkflowAlreadyComplete:
            logger.info('Workflow already complete.')

#        broken_locations = get_broken_locations(data_interface)
#        data_interface.save_broken_locations_report(broken_locations)

#        report = make_broken_location_report(broken_locations)
#        if report:
#            logger.warning(report)

    logger.info(f'Fit version {specification.data.output_root} complete.')


def get_broken_locations(data_interface: FitDataInterface):
    _runner = functools.partial(
        _get_broken_locations,
        data_interface=data_interface,
    )
    results = parallel.run_parallel(
        _runner,
        arg_list=list(range(data_interface.get_n_draws())),
        num_cores=26,
        progress_bar=True
    )

    hierarchy = data_interface.load_hierarchy('pred')
    name_map = hierarchy.set_index('location_id')['location_name']

    below_0, over_total_pop = zip(*results)
    below_0 = pd.concat(below_0, axis=1)
    over_total_pop = pd.concat(over_total_pop, axis=1)

    reports = []
    for location_id in below_0.index.tolist():
        loc_below_0, loc_over_total_pop = below_0.loc[location_id], over_total_pop.loc[location_id]

        loc_report = {
            'missing': _extract_draws(loc_below_0, loc_below_0.isnull()),
            'below_0': _extract_draws(loc_below_0, loc_below_0.fillna(False)),
            'over_total_pop': _extract_draws(loc_over_total_pop, loc_over_total_pop.fillna(False)),
        }
        loc_report['any_error'] = sorted(list(set().union(*loc_report.values())))
        if not loc_report['any_error']:
            continue

        loc_report['location_id'] = location_id
        loc_report['location_name'] = name_map.loc[location_id]
        reports.append(loc_report)
    return reports


def _get_broken_locations(draw_id: int,
                          data_interface: FitDataInterface):
    total_population = data_interface.load_population(measure='total').population
    infections = data_interface.load_posterior_epi_measures(
        draw_id, columns=['daily_naive_unvaccinated_infections', 'round']
    )
    infections = infections.loc[infections['round'] == 2, 'daily_naive_unvaccinated_infections']

    below_0 = (infections.groupby('location_id').min() < 0).rename(f'draw_{draw_id}')

    over_total_pop = infections.groupby('location_id').sum()
    over_total_pop = ((over_total_pop / total_population.reindex(over_total_pop.index)) > 1).rename(f'draw_{draw_id}')

    return below_0, over_total_pop


def _extract_draws(data: pd.Series, mask: pd.Series):
    return sorted([int(draw.split('_')[1]) for draw in data[mask].index.tolist()])


def _make_loc_issue_report(key: str, problem_draws: List[str]):
    if not problem_draws:
        return ''
    loc_issue_report = f'    {key}: {len(problem_draws)} ['
    for i, problem_draw in enumerate(problem_draws):
        if i < 3:
            loc_issue_report += f'{problem_draw}, '
        else:
            loc_issue_report += f'..., '
            break
    loc_issue_report = loc_issue_report[:-2] + ']\n'
    return loc_issue_report


def make_broken_location_report(broken_locations: List[Dict]):
    must_drop = []
    should_drop = []
    should_drop_threshold = 5
    report = ''
    for location_data in broken_locations:
        report += f'{location_data["location_name"]} ({location_data["location_id"]}):\n'
        for key in ['any_error', 'missing', 'below_0', 'over_total_pop']:
            report += _make_loc_issue_report(key, location_data[key])

        if location_data['missing']:
            must_drop.append(location_data['location_id'])
        elif len(location_data['any_error']) > should_drop_threshold:
            should_drop.append(location_data['location_id'])

    if report:
        report = '\n' + report
    if must_drop:
        report += f'must_drop_locations (draws are missing): {must_drop}\n'
    if should_drop:
        report += f'should_drop_locations (more than {should_drop_threshold} draws broken): {should_drop}\n'

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
