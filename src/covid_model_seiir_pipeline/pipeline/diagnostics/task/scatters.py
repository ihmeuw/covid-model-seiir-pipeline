from pathlib import Path

import click
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    static_vars,
)
from covid_model_seiir_pipeline.pipeline.diagnostics.specification import (
    DiagnosticsSpecification,
)


logger = cli_tools.task_performance_logger


def run_scatters(diagnostics_version: str, name: str) -> None:
    logger.info(f'Starting scatters for version {diagnostics_version}, name {name}.', context='setup')
    diagnostics_spec = DiagnosticsSpecification.from_path(
        Path(diagnostics_version) / static_vars.DIAGNOSTICS_SPECIFICATION_FILE
    )
    scatters_spec = [spec for spec in diagnostics_spec.scatters if spec.name == name].pop()

    logger.report()


@click.command()
@cli_tools.with_task_diagnostics_version
@cli_tools.with_name
@cli_tools.add_verbose_and_with_debugger
def scatters(diagnostics_version: str, name: str,
             verbose: int, with_debugger: bool):
    """Produce scatters corresponding to the configuration associated with NAME"""
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_scatters, logger, with_debugger)
    run(diagnostics_version=diagnostics_version,
        name=name)


if __name__ == '__main__':
    scatters()
