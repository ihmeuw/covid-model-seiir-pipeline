from typing import List, Tuple, Dict
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.model.epi_measures import (
    aggregate_data_from_md,
)


def condition_out_variants(sero_location_dates: List[Tuple[int, pd.Timestamp]],
                           daily_infections: pd.Series,
                           hierarchy: pd.DataFrame,
                           variant_prevalence: pd.Series,
                           variant_risk_ratios: pd.DataFrame,
                           exposure_to_seroconversion: int,
                           seroconversion_to_measure: int):
    variant_infections = variant_prevalence.drop(['omega'], axis=1).multiply(daily_infections, axis=0)

    # agg infections and RR
    offset = 1e-4
    variant_risk_ratios = (variant_infections + offset) * variant_risk_ratios.loc[:, variant_infections.columns]
    variant_risk_ratios = pd.concat([
        aggregate_data_from_md(variant_risk_ratios.loc[:, variant].reset_index(),
                               hierarchy, variant).set_index(['location_id', 'date'])
        for variant in variant_risk_ratios.columns
    ], axis=1).sort_index()
    variant_infections = pd.concat([
        aggregate_data_from_md(variant_infections.loc[:, variant].reset_index(),
                               hierarchy, variant).set_index(['location_id', 'date'])
        for variant in variant_infections.columns
    ], axis=1).sort_index()
    variant_risk_ratios /= (variant_infections + offset)

    # get cumulative variant prevalence
    variant_infections = (variant_infections
                          .dropna(how='all')
                          .groupby('location_id').cumsum())
    variant_infections = variant_infections.loc[variant_infections.sum(axis=1) > 0]
    variant_infections = variant_infections.divide(variant_infections.sum(axis=1), axis=0)
    
    # get total cumulative risk ratio by day
    ratio_data_scalar = (variant_infections * variant_risk_ratios).sum(axis=1)
    ratio_data_scalar = (ratio_data_scalar
                         .groupby('location_id').shift(exposure_to_seroconversion)
                         .loc[sero_location_dates]
                         .dropna()
                         .rename('ratio_data_scalar'))
    ratio_data_scalar = ratio_data_scalar.reset_index('date')
    if seroconversion_to_measure < 0:
        ratio_data_scalar['date'] += pd.Timedelta(days=seroconversion_to_measure)
    else:
        ratio_data_scalar['date'] -= pd.Timedelta(days=-seroconversion_to_measure)
    ratio_data_scalar = ratio_data_scalar.set_index('date', append=True).sort_index()

    # value is effect of variant, invert to become multiplicative scalar (want to remove that effect)
    ratio_data_scalar = 1 / ratio_data_scalar

    return ratio_data_scalar
