from typing import Dict, List

import pandas as pd
from loguru import logger

from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters
from covid_model_seiir_pipeline.pipeline.fit.model import (
    idr,
    ifr,
    ihr,
)
from covid_model_seiir_pipeline.pipeline.fit.model.sampled_params import (
    Durations,
    VariantRR,
)


def run_rates_pipeline(epi_data: pd.DataFrame,
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
                       progress_bar: bool):
    logger.debug('IDR ESTIMATION')
    idr_results = idr.runner(
        cumulative_cases=epi_data['cumulative_cases'].dropna(),
        daily_cases=epi_data['daily_cases'].dropna(),
        seroprevalence=seroprevalence,
        covariates=covariates,
        covariate_list=covariate_pool['idr'],
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        population=total_population,
        testing_capacity=testing_capacity,
        daily_infections=daily_infections,
        durations=durations._asdict(),
        pred_start_date=params.pred_start_date,
        pred_end_date=params.pred_end_date,
        num_threads=num_threads,
        progress_bar=progress_bar,
    )

    logger.debug('IHR ESTIMATION')
    ihr_results = ihr.runner(
        cumulative_hospitalizations=epi_data['cumulative_hospitalizations'].dropna(),
        daily_hospitalizations=epi_data['daily_hospitalizations'].dropna(),
        seroprevalence=seroprevalence,
        covariates=covariates,
        covariate_list=covariate_pool['ihr'],
        daily_infections=daily_infections,
        variant_prevalence=variant_prevalence,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        ihr_age_pattern=age_patterns['ihr'],
        sero_age_pattern=age_patterns['seroprevalence'],
        population=total_population,
        age_spec_population=age_specific_population,
        variant_risk_ratio=variant_rrs.ihr,
        durations=durations._asdict(),
        day_0=params.day_0,
        pred_start_date=params.pred_start_date,
        pred_end_date=params.pred_end_date,
    )

    logger.debug("IFR ESTIMATION")
    ifr_results = ifr.runner(
        cumulative_deaths=epi_data['cumulative_deaths'].dropna(),
        daily_deaths=epi_data['daily_deaths'].dropna(),
        seroprevalence=seroprevalence,
        covariates=covariates,
        covariate_list=covariate_pool['ifr'],
        daily_infections=daily_infections,
        variant_prevalence=variant_prevalence,
        mr_hierarchy=mr_hierarchy,
        pred_hierarchy=pred_hierarchy,
        ifr_age_pattern=age_patterns['ifr'],
        sero_age_pattern=age_patterns['seroprevalence'],
        population=total_population,
        age_spec_population=age_specific_population,
        variant_risk_ratio=variant_rrs.ifr,
        durations=durations._asdict(),
        day_inflection=day_inflection,
        day_0=params.day_0,
        pred_start_date=params.pred_start_date,
        pred_end_date=params.pred_end_date,
    )

    return ifr_results, ihr_results, idr_results
