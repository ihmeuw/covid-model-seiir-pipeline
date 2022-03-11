from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger

from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters
from covid_model_seiir_pipeline.pipeline.fit.model.rates import (
    idr,
    ifr,
    ihr,
)
from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
    VariantRR,
)


def run_rates_pipeline(measure: str,
                       epi_data: pd.DataFrame,
                       age_patterns: pd.DataFrame,
                       seroprevalence: pd.DataFrame,
                       covariates: List[pd.Series],
                       covariate_pool: Dict[str, List[str]],
                       mr_hierarchy: pd.DataFrame,
                       pred_hierarchy: pd.DataFrame,
                       total_population: pd.Series,
                       age_specific_population: pd.Series,
                       testing_capacity: pd.Series,
                       variant_prevalence: pd.Series,
                       daily_infections: pd.Series,
                       durations: Durations,
                       variant_rrs: VariantRR,
                       params: RatesParameters,
                       day_inflection: pd.Timestamp,
                       num_threads: int,
                       progress_bar: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runner = {
        'case': idr.runner,
        'admission': ihr.runner,
        'death': ifr.runner,
    }[measure]

    logger.debug('RATES ESTIMATION')
    results, data = runner(
        epi_data=epi_data,
        seroprevalence=seroprevalence,
        covariates=covariates,
        covariate_pool=covariate_pool,
        daily_infections=daily_infections,
        variant_prevalence=variant_prevalence,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        age_patterns=age_patterns,
        population=total_population,
        age_specific_population=age_specific_population,
        variant_risk_ratio=variant_rrs._asdict(),
        testing_capacity=testing_capacity,
        durations=durations._asdict(),
        pred_start_date=params.pred_start_date,
        pred_end_date=params.pred_end_date,
        day_inflection=day_inflection,
        day_0=params.day_0,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )
    return results, data
