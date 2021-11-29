import itertools

import click

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    utilities,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification

logger = cli_tools.task_performance_logger


def run_covariate_pool(fit_version: str) -> None:
    logger.info('Starting beta fit.', context='setup')
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    n_samples = data_interface.get_n_draws()

    logger.info('Loading covariate data', context='read')
    # Get from config?
    covariate_options = [
        'obesity',
        'smoking',
        'diabetes',
        'ckd',
        'cancer',
        'copd',
        'cvd',
        'uhc',
        'haq',
        'prop_65plus'
    ]

    logger.info('Identifying best covariate combinations and inflection points.', context='model')
    test_combinations = []
    for i in range(len(covariate_options)):
        test_combinations += [list(set(cc)) for cc in itertools.combinations(covariate_options, i + 1)]
    test_combinations = [cc for cc in test_combinations if
                         len([c for c in cc if c in ['uhc', 'haq']]) <= 1]
    logger.warning('Not actually testing covariate combinations.')
    selected_combinations = [tc for tc in test_combinations if 'smoking' in tc and len(tc) >= 5][:n_samples]

    idr_covariate_options = [['haq'], ['uhc'], ['prop_65plus'], [], ]
    random_state = utilities.get_random_state('idr_covariate_pool')
    idr_covariate_pool = random_state.choice(idr_covariate_options, n_samples)

    covariate_selections = {'ifr': {}, 'ihr': {}, 'idr': {}}
    for draw in range(n_samples):
        covariate_selections['ifr'][draw] = selected_combinations[draw]
        covariate_selections['ihr'][draw] = selected_combinations[draw]
        covariate_selections['idr'][draw] = idr_covariate_pool[draw]

    logger.info('Writing covariate options', context='write')
    data_interface.save_covariate_options(covariate_selections)

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.add_verbose_and_with_debugger
def covariate_pool(fit_version: str,
                   verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_covariate_pool, logger, with_debugger)
    run(fit_version=fit_version)
