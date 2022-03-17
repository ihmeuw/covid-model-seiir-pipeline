from pathlib import Path
from typing import Callable

import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model import (
    helpers,
)

logger = cli_tools.task_performance_logger


def preprocess_mask_use(data_interface: PreprocessingDataInterface) -> None:
    for scenario in ['reference', 'best', 'worse']:
        logger.info(f'Loading mask use data for scenario {scenario}.', context='read')
        mask_use = data_interface.load_raw_mask_use_data(scenario)

        logger.info(f'Writing mask use data for scenario {scenario}.', context='write')
        data_interface.save_covariate(mask_use, 'mask_use', scenario)


def preprocess_prop_65plus(data_interface: PreprocessingDataInterface) -> None:
    logger.info('Loading population data', context='read')
    pop = data_interface.load_population('five_year').reset_index()
    hierarchy = data_interface.load_hierarchy('pred')

    logger.info('Generating prop 65+', context='transform')
    over_65 = pop[pop.age_group_years_start >= 65].groupby('location_id').population.sum()
    total = pop.groupby('location_id').population.sum()
    prop_65plus = (over_65 / total).rename('prop_65plus_reference')
    prop_65plus = helpers.parent_inheritance(prop_65plus, hierarchy)

    logger.info('Writing covariate', context='write')
    data_interface.save_covariate(prop_65plus, 'prop_65plus', 'reference')


def preprocess_mobility(data_interface: PreprocessingDataInterface) -> None:
    hierarchy = data_interface.load_hierarchy('pred')
    for scenario in ['reference', 'vaccine_adjusted']:
        logger.info(f'Loading raw mobility data for scenario {scenario}.', context='read')
        mobility = data_interface.load_raw_mobility(scenario)
        percent_mandates = data_interface.load_raw_percent_mandates(scenario)

        logger.info(f'Fixing southern hemisphere locations.', context='transform')
        percent_mandates = _adjust_southern_hemisphere(percent_mandates)
        mobility = helpers.parent_inheritance(mobility, hierarchy)
        percent_mandates = helpers.parent_inheritance(percent_mandates, hierarchy)

        logger.info(f'Writing mobility data for scenario {scenario}.', context='write')
        data_interface.save_covariate(mobility, 'mobility', scenario)
        data_interface.save_covariate_info(percent_mandates, 'mobility', f'percent_mandates_{scenario}')

    logger.info('Loading raw mobility effect sizes.', context='read')
    effect_sizes = data_interface.load_raw_mobility_effect_sizes()

    logger.info('Saving mobility effect sizes.', context='write')
    data_interface.save_covariate_info(effect_sizes, 'mobility', 'effect')


def _adjust_southern_hemisphere(data: pd.DataFrame) -> pd.DataFrame:
    # adjust the mandate lift so s hemisphere locs don't have duplicated rows
    # Hack from Haley:
    # percent_mandates has a value when observed, so use that to subset out duplicated
    s_hemisphere_locs = data.loc[data['percent_mandates'].notnull(), 'location_id'].unique()
    in_s_hemisphere = data['location_id'].isin(s_hemisphere_locs)
    first_observed = (data[in_s_hemisphere]
                      .groupby('location_id')['date']
                      .min())
    first_predicted = (data[in_s_hemisphere & data['percent_mandates'].isnull()]
                       .groupby('location_id')['date']
                       .min())
    date_shift = (first_predicted - first_observed).rename('shift').reset_index()

    # Subset to just the projected time series
    data = data.loc[data['percent_mandates'].isnull()]
    data = data.merge(date_shift, how='outer', left_on='location_id', right_on='location_id')
    data.loc[in_s_hemisphere, 'date'] = (
            data.loc[in_s_hemisphere, 'date'] - data.loc[in_s_hemisphere, 'shift']
    )
    return data


def preprocess_pneumonia(data_interface: PreprocessingDataInterface) -> None:
    logger.info(f'Loading raw pneumonia data', context='read')
    data = data_interface.load_raw_pneumonia_data()
    hierarchy = data_interface.load_hierarchy('pred')

    logger.info('Extending pneumonia pattern into the future.', context='transform')
    next_year = data.copy()
    next_year['date'] += pd.Timedelta(days=366)
    next_year = next_year.groupby("location_id", as_index=False).apply(lambda x: x.iloc[1:-1]).reset_index(drop=True)
    year_after_next = next_year.copy()
    year_after_next['date'] += pd.Timedelta(days=365)
    year_after_that = year_after_next.copy()
    year_after_that['date'] += pd.Timedelta(days=365)
    data = (pd.concat([data, next_year, year_after_next, year_after_that])
            .sort_values(["location_id", "date"])
            .reset_index(drop=True))
    data = helpers.parent_inheritance(data, hierarchy)

    logger.info(f'Writing pneumonia data.', context='write')
    data_interface.save_covariate(data, 'pneumonia', 'reference')


def preprocess_population_density(data_interface: PreprocessingDataInterface) -> None:
    logger.info(f'Loading raw population density data', context='read')
    data = data_interface.load_raw_population_density_data()
    hierarchy = data_interface.load_hierarchy('pred')

    logger.info('Computing population density aggregate.', context='transform')
    exclude_columns = [
        '<150 ppl/sqkm',
        '150-300 ppl/sqkm',
        '300-500 ppl/sqkm',
        '500-1000 ppl/sqkm',
        '1000-2500 ppl/sqkm',
    ]
    data = (data
            .drop(columns=exclude_columns)
            .sum(axis=1)
            .reset_index()
            .rename(columns={0: 'proportion_over_2_5k_reference'}))
    data['observed'] = float('nan')
    data = helpers.parent_inheritance(data, hierarchy)

    logger.info(f'Writing population density covariate.', context='write')
    data_interface.save_covariate(data, 'proportion_over_2_5k', 'reference')


def preprocess_testing_data(data_interface: PreprocessingDataInterface) -> None:
    logger.info('Loading raw testing data', context='read')
    data = data_interface.load_raw_testing_data()
    hierarchy = data_interface.load_hierarchy('pred')
    hk = data[data.location_id == 354]
    hk['location_id'] == 44533
    data = pd.concat([data[~(data.location_id == 354)], hk]).sort_values(['location_id', 'date'])

    logger.info('Processing testing for IDR calc and beta covariate', context='transform')
    testing_for_idr = _process_testing_for_idr(data.copy())
    testing_for_beta = (data
                        .rename(columns={'test_pc': 'testing_reference'})
                        .loc[:, ['location_id', 'date', 'observed', 'testing_reference']])
    testing_for_beta = helpers.parent_inheritance(testing_for_beta, hierarchy)

    logger.info('Writing testing data', context='write')
    data_interface.save_testing_data(testing_for_idr)
    data_interface.save_covariate(testing_for_beta, 'testing', 'reference')


def _process_testing_for_idr(data: pd.DataFrame) -> pd.DataFrame:
    data['population'] = data['population'].fillna(data['pop'])
    data['daily_tests'] = data['test_pc'] * data['population']
    data['cumulative_tests'] = data.groupby('location_id')['daily_tests'].cumsum()
    data = (data
            .loc[:, ['location_id', 'date', 'cumulative_tests']]
            .sort_values(['location_id', 'date'])
            .reset_index(drop=True))
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: helpers.fill_dates(x, ['cumulative_tests']))
            .reset_index(drop=True))
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['daily_tests'] = (data
                           .groupby('location_id')['cumulative_tests']
                           .apply(lambda x: x.diff()))
    data = data.dropna()
    data = data.sort_values(['location_id', 'date']).reset_index(drop=True)
    data['testing_capacity'] = data.groupby('location_id')['daily_tests'].cummax()

    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['daily_tests', 'testing_capacity', 'cumulative_tests']])
    return data


def preprocess_variant_prevalence(data_interface: PreprocessingDataInterface) -> None:
    hierarchy = data_interface.load_hierarchy('pred')
    for scenario in ['reference']:
        logger.info(f'Loading raw variant prevalence data for scenario {scenario}.', context='read')
        data = data_interface.load_raw_variant_prevalence(scenario)

        logger.info('Parsing into WHO variant of concern.', context='transform')
        data = _process_variants_of_concern(data)
        invasion_dates = data[data.omicron > 0.01].groupby('location_id').date.min()
        data = data.set_index(['location_id', 'date'])

        logger.info(f'Overwriting invasion dates based on case inflection point.', context='replace')
        p = Path(__file__).parent / 'invasion_date_hardcodes.csv'
        target_dates = pd.read_csv(p).set_index('location_id')
        target_dates['target_date'] = (pd.to_datetime(target_dates['case_inflection_date'])
                                       .fillna(pd.to_datetime(target_dates['data_date']) + pd.Timedelta(days=7)))
        target_dates = target_dates.apply(lambda x: x['target_date'] - pd.Timedelta(days=x['lag']), axis=1)
        target_dates = target_dates.loc[invasion_dates.reset_index()['location_id'].unique()]
        shifts = (target_dates - invasion_dates.loc[target_dates.index])
        shifts = shifts[shifts != pd.Timedelta(days=0)].dt.days.to_dict()

        updates = []
        for location_id, invasion_date in shifts.items():
            old_omicron = data.loc[location_id, 'omicron']
            new_omicron = old_omicron.shift(invasion_date).ffill().bfill()
            new_data = data.loc[location_id].drop(columns='omicron')
            new_data['omicron'] = new_omicron
            new_data = new_data.div(new_data.sum(axis=1), axis=0)
            new_data['location_id'] = location_id
            updates.append(new_data.reset_index().set_index(['location_id', 'date']))
        updates = pd.concat(updates)
        data = data.drop(updates.index).append(updates).sort_index()
        data = helpers.parent_inheritance(data, hierarchy)
        delhi_variant_level = data.loc[4849]
        dfs = []
        for location_id in [4840, 4845, 60896, 4858, 4866]:
            df = delhi_variant_level.copy()
            df['location_id'] = location_id
            df = df.reset_index().set_index(['location_id', 'date'])
            dfs.append(df)
        data = pd.concat([data] + dfs).sort_index()
        logger.info(f'Writing {scenario} scenario data.', context='write')
        data_interface.save_variant_prevalence(data, scenario)


def _process_variants_of_concern(data: pd.DataFrame) -> pd.DataFrame:
    variant_map = {
        'alpha': ['B117'],
        'beta': ['B1351'],
        'gamma': ['P1'],
        'delta': ['B16172'],
        'omicron': ['Omicron'],
        'other': ['B1621', 'C37'],
        'ancestral': ['wild_type'],
    }
    drop = []
    all_variants = [lineage for lineages in variant_map.values() for lineage in lineages] + drop
    missing_variants = set(all_variants).difference(data.columns)
    if missing_variants:
        raise ValueError(f'Missing variants {missing_variants}')
    extra_variants = set(data.columns).difference(all_variants)
    if extra_variants:
        raise ValueError(f'Unknown variants {extra_variants}')
    if drop:
        logger.warning(f'Dropping variants: {drop}')

    for var_name, lineages in variant_map.items():
        data[var_name] = data[lineages].sum(axis=1)
    data = data[list(variant_map)]
    data.loc[:, 'omega'] = 0.
    data.loc[data.sum(axis=1) == 0, 'ancestral'] = 1
    if (data.sum(axis=1) < 1 - 1e-5).any():
        raise ValueError("Variant prevalence sums to less than 1 for some location-dates.")
    if (data.sum(axis=1) > 1 + 1e-5).any():
        raise ValueError("Variant prevalence sums to more than 1 for some location-dates.")

    return data.reset_index()


def preprocess_gbd_covariate(covariate: str) -> Callable[[PreprocessingDataInterface], None]:

    def _preprocess_gbd_covariate(data_interface: PreprocessingDataInterface) -> None:
        logger.info(f'Loading {covariate} data for.', context='read')
        hierarchy = data_interface.load_hierarchy('pred')
        data = data_interface.load_gbd_covariate(covariate)

        if covariate in ['uhc', 'haq']:
            data[f'{covariate}_reference'] /= 100

        data = helpers.parent_inheritance(data, hierarchy)

        logger.info(f'Writing {covariate} data.', context='write')
        data_interface.save_covariate(data, covariate, 'reference')

    return _preprocess_gbd_covariate
