import click
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.lib.ode_mk2.constants import (
    VARIANT_NAMES,
    RISK_GROUP_NAMES,
    EPI_MEASURE_NAMES,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification
from covid_model_seiir_pipeline.pipeline.fit import model

logger = cli_tools.task_performance_logger


def run_covariate_pool(fit_version: str, progress_bar: bool) -> None:
    logger.info('Starting beta fit.', context='setup')
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)

    logger.info('Identifying best covariate combinations and inflection points.', context='model')
    covariate_options = ['obesity', 'smoking', 'diabetes', 'ckd',
                         'cancer', 'copd', 'cvd', 'uhc', 'haq',]
    covariates = [db.obesity(adj_gbd_hierarchy),
                  db.smoking(adj_gbd_hierarchy),
                  db.diabetes(adj_gbd_hierarchy),
                  db.ckd(adj_gbd_hierarchy),
                  db.cancer(adj_gbd_hierarchy),
                  db.copd(adj_gbd_hierarchy),
                  db.cvd(adj_gbd_hierarchy),
                  db.uhc(adj_gbd_hierarchy) / 100,
                  db.haq(adj_gbd_hierarchy) / 100,]
    prop_65plus = age_spec_population.copy().reset_index()
    prop_65plus = prop_65plus.loc[prop_65plus['age_group_years_start'] >= 65].groupby('location_id')['population'].sum() /\
                  prop_65plus.groupby('location_id')['population'].sum()
    prop_65plus = prop_65plus.rename('prop_65plus')
    covariates += [prop_65plus.copy()]
    del prop_65plus
    test_combinations = []
    for i in range(len(covariate_options)):
        test_combinations += [list(set(cc)) for cc in itertools.combinations(covariate_options, i + 1)]
    test_combinations = [cc for cc in test_combinations if
                         len([c for c in cc if c in ['uhc', 'haq']]) <= 1]
    logger.warning('Not actually testing covariate combinations.')
    selected_combinations = [tc for tc in test_combinations if 'smoking' in tc and len(tc) >= 5][:n_samples]
    # selected_combinations = covariate_selection.covariate_selection(
    #     n_samples=n_samples, test_combinations=test_combinations,
    #     model_inputs_root=model_inputs_root, excess_mortality=excess_mortality,
    #     shared=shared,
    #     reported_seroprevalence=reported_seroprevalence,
    #     covariate_options=covariate_options,
    #     covariates=covariates,
    #     cutoff_pct=1.,
    #     durations={'sero_to_death': int(round(np.mean(durations.EXPOSURE_TO_ADMISSION) + \
    #                                           np.mean(durations.ADMISSION_TO_DEATH) - \
    #                                           np.mean(durations.EXPOSURE_TO_SEROCONVERSION)))
    #               },
    # )

    idr_covariate_options = [['haq'], ['uhc'], ['prop_65plus'], [], ]
    random_state = get_random_state('idr_covariate_pool')
    idr_covariate_pool = random_state.choice(idr_covariate_options, n_samples, )

    logger.report()


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.with_draw_id
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def covariate_pool(fit_version: str,
                   progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_covariate_pool, logger, with_debugger)
    run(fit_version=fit_version,
        progress_bar=progress_bar)
