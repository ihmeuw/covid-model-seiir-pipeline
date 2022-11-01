import pandas as pd

from covid_model_seiir_pipeline.pipeline.forecasting.model.containers import Indices


def get_reimposition_threshold(
        indices: Indices,
        population: pd.Series,
        epi_data: pd.DataFrame,
        rhos: pd.DataFrame,
        mortality_scalars: pd.DataFrame,
        hierarchy: pd.DataFrame,
        scalar: float = 5.0,
        threshold_measure: str = 'deaths',
        min_threshold_rate: float = 10.0,
        max_threshold_rate: float = 30.0,
):
    past_measure = epi_data.loc[indices.past, f'smoothed_daily_{threshold_measure}'].rename('threshold_measure')

    omicron_prevalence = rhos[['omicron', 'ba5']].sum(axis=1)

    omicron_invasion = (omicron_prevalence[omicron_prevalence > 0.]
                        .reset_index()
                        .groupby('location_id')
                        .date
                        .min()
                        .rename('omicron_invasion')
                        .reindex(past_measure.index, level='location_id'))

    data = pd.concat([past_measure, omicron_invasion], axis=1).reset_index()
    max_omicron_measure = (data[data.date > data.omicron_invasion]
                           .groupby('location_id')
                           .threshold_measure
                           .max()
                           .reindex(past_measure.reset_index().location_id.unique(),
                                    fill_value=0.0)
                           .sort_index())

    pop = population.loc[max_omicron_measure.index]
    mortality_scalars = (mortality_scalars
                         .groupby('location_id')
                         .mean()
                         .loc[max_omicron_measure.index])

    min_rate = min_threshold_rate * mortality_scalars
    max_rate = max_threshold_rate * mortality_scalars
    raw_rate = scalar * max_omicron_measure / pop * 1_000_00
    threshold_rate = raw_rate.clip(min_rate, max_rate).rename('threshold_rate')

    china = hierarchy.path_to_top_parent.str.contains(',6,')
    china_subnats = hierarchy[china & (hierarchy.most_detailed == 1)].location_id.tolist()
    threshold_rate.loc[china_subnats] = 1

    return threshold_rate


def compute_reimposition_dates(
    compartments: pd.DataFrame,
    total_population: pd.Series,
    min_reimposition_dates: pd.Series,
    reimposition_threshold: pd.Series,
):
    daily_death_rate = compartments.filter(like='Death_all_all_all').sum(
        axis=1).diff() / total_population * 1_000_000

    reimposition_dates = []
    for location_id in daily_death_rate.reset_index().location_id.unique():
        start_date = min_reimposition_dates.loc[location_id]
        threshold = reimposition_threshold.loc[location_id]
        loc_ddr = daily_death_rate.loc[location_id].loc[start_date:]
        above_threshold = loc_ddr[loc_ddr > threshold]
        if not above_threshold.empty:
            date = above_threshold.reset_index().date.min()
            reimposition_dates.append((location_id, date))
    reimposition_dates = (pd.DataFrame(reimposition_dates, columns=['location_id', 'date'])
                          .set_index('location_id')
                          .date)
    return reimposition_dates


def reimpose_mandates(
    reimposition_dates: pd.Series,
    covariates: pd.DataFrame,
):
    import pdb; pdb.set_trace()

