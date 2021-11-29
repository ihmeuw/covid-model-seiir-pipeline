import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.specification import RatesParameters


def subset_seroprevalence(seroprevalence: pd.DataFrame,
                          epi_data: pd.DataFrame,
                          variant_prevalence: pd.Series,
                          population: pd.Series,
                          params: RatesParameters) -> pd.DataFrame:
    daily_deaths = epi_data['daily_deaths'].dropna()

    death_dates = (daily_deaths.loc[(daily_deaths / population) < params.death_rate_threshold]
                   .reset_index()
                   .groupby('location_id')['date'].max()
                   .rename('death_date'))
    variant_dates = (variant_prevalence.loc[variant_prevalence < params.variant_prevalence_threshold]
                     .reset_index()
                     .groupby('location_id')['date'].max()
                     .rename('variant_date'))
    invasion_dates = pd.concat([death_dates, variant_dates], axis=1)
    invasion_dates = (invasion_dates
                      .fillna(invasion_dates.max().max())
                      .min(axis=1)
                      .rename('invasion_date'))
    inclusion_date = ((invasion_dates + pd.Timedelta(days=params.inclusion_days))
                      .rename('inclusion_date')
                      .reset_index())
    seroprevalence = seroprevalence.merge(inclusion_date)
    seroprevalence = seroprevalence.loc[seroprevalence['date'] <= seroprevalence['inclusion_date']]
    del seroprevalence['inclusion_date']

    return seroprevalence
