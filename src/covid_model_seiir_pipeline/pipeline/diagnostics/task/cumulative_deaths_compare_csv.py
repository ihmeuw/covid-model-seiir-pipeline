import itertools
from pathlib import Path

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.postprocessing import (
    PostprocessingSpecification,
    PostprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    DiagnosticsSpecification,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.model.pdf_merger import (
    get_locations_dfs,
)

logger = cli_tools.task_performance_logger


def run_cumulative_deaths_compare_csv(diagnostics_version: str) -> None:
    """Make the grid plots!"""
    logger.info(f'Starting cumulative death compare csv for version {diagnostics_version}', context='setup')
    diagnostics_spec = DiagnosticsSpecification.from_path(
        Path(diagnostics_version) / static_vars.DIAGNOSTICS_SPECIFICATION_FILE
    )
    cumulative_death_spec = diagnostics_spec.cumulative_deaths_compare_csv

    postprocessing_spec = PostprocessingSpecification.from_path(
        Path(cumulative_death_spec.comparators[-1].version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
    )
    postprocessing_data_interface = PostprocessingDataInterface.from_specification(postprocessing_spec)
    hierarchy = postprocessing_data_interface.load_hierarchy()
    sorted_locs = get_locations_dfs(hierarchy)

    data = []
    max_date = pd.Timestamp('2000-01-01')

    for comparator in cumulative_death_spec.comparators:
        postprocessing_spec = PostprocessingSpecification.from_path(
            Path(comparator.version) / static_vars.POSTPROCESSING_SPECIFICATION_FILE
        )
        postprocessing_data_interface = PostprocessingDataInterface.from_specification(postprocessing_spec)
        full_data = postprocessing_data_interface.load_full_data()
        max_date = max(max_date, full_data.reset_index().date.max())

        for (scenario, label), measure in itertools.product(comparator.scenarios.items(), ['deaths', 'unscaled_deaths']):
            df = postprocessing_data_interface.load_output_summaries(scenario, measure=f'cumulative_{measure}')
            suffix = '_unscaled' if 'unscaled' in measure else ''
            df = df['mean'].rename(f'{label}{suffix}')
            data.append(df)
    data = pd.concat(data, axis=1).loc[sorted_locs].reset_index()
    data['location_name'] = data.location_id.map(hierarchy.set_index('location_id').location_ascii_name)

    dates = cumulative_death_spec.dates + [str(max_date.date)]
    final_data = []
    for date in dates:
        date_data = data[data.date == pd.Timestamp(date)].set_index(['location_id', 'location_name'])
        date_data.columns = [f'{date}_{c}'for c in date_data.columns]
        final_data.append(date_data)
    final_data = pd.concat(final_data, axis=1)
    import pdb; pdb.set_trace()
    logger.report()


@click.command()
@cli_tools.with_task_diagnostics_version
@cli_tools.add_verbose_and_with_debugger
def cumulative_deaths_compare_csv(diagnostics_version: str,
                                  verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_cumulative_deaths_compare_csv, logger, with_debugger)
    run(diagnostics_version=diagnostics_version)


if __name__ == '__main__':
    cumulative_deaths_compare_csv()
