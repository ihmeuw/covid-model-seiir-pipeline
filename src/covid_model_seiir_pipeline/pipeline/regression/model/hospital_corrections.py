from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.regression.model.containers import (
    HospitalFatalityRatioData,
    HospitalCensusData,
    HospitalMetrics,
    HospitalCorrectionFactors,
)

if TYPE_CHECKING:
    from covid_model_seiir_pipeline.pipeline.regression.specification import (
        HospitalParameters,
    )


# FIXME: These bins are a hangover from early covid CDC data.
AGE_BINS = [0, 55, 65, 75, 85, 125]


def get_death_weights(mr: pd.Series, population: pd.DataFrame, with_error: bool) -> pd.Series:
    """Get a series of weights to split all age deaths into age-specific."""
    pop = population.copy()
    # Coerce population to series matching index of MR
    pop['age_start'] = pop['age_group_years_start'].astype(int)
    pop = pop.loc[:, ['location_id', 'age_start', 'population']].set_index(['location_id', 'age_start']).population

    calc_func = {True: _get_death_weight_error, False: _get_death_weight_correct}[with_error]
    return calc_func(mr, pop)


def _get_death_weight_correct(mr: pd.Series, pop: pd.Series) -> pd.Series:
    def _compute_death_weight(df, start, end):
        return df.loc[(start <= df.age_start) & (df.age_start < end), 'mr_prob'].sum() / df['mr_prob'].sum()

    mr_prob = (mr * pop).rename('mr_prob').reset_index()
    death_weight_dfs = []
    for age_start, age_end in zip(AGE_BINS, AGE_BINS[1:]):
        age_death_weight = (mr_prob
                            .groupby('location_id')
                            .apply(lambda x: _compute_death_weight(x, age_start, age_end))
                            .rename('death_weight')
                            .reset_index())
        age_death_weight['age'] = age_start + (age_end - age_start) / 2
        death_weight_dfs.append(age_death_weight)
    death_weights = pd.concat(death_weight_dfs)

    return death_weights.set_index(['location_id', 'age']).sort_index().death_weight


def _get_death_weight_error(mr: pd.Series, pop: pd.Series) -> pd.Series:
    combined_mr_pop = pd.concat([mr, pop], axis=1)

    def _compute_mr_prob_error(df, start, end):
        df = df.reset_index()
        this_bin_bad = df[(start <= df.age_start) & (df.age_start < end - 5)]
        bad_weighted_mr = (this_bin_bad['MRprob'] * this_bin_bad['population'] / this_bin_bad['population'].sum()).sum()
        this_actual_pop = df.loc[(age_start <= df.age_start) & (df.age_start < age_end), 'population'].sum()
        return bad_weighted_mr * this_actual_pop

    weighted_mr_dfs = []
    for age_start, age_end in zip(AGE_BINS, AGE_BINS[1:]):
        mr_weight = (combined_mr_pop
                     .groupby('location_id')
                     .apply(lambda x: _compute_mr_prob_error(x, age_start, age_end))
                     .rename('bad_weighted_mr')
                     .reset_index())
        mr_weight['age'] = age_start + (age_end - age_start) / 2
        weighted_mr_dfs.append(mr_weight)
    bad_weighted_mr = pd.concat(weighted_mr_dfs).set_index(['location_id', 'age']).sort_index().bad_weighted_mr
    bad_death_weights = (bad_weighted_mr / bad_weighted_mr.groupby('location_id').sum()).rename('death_weight')
    return bad_death_weights


def _bound(low, high, value):
    """Helper to fix out of bounds probabilities.
    As written, we can have a negative probability of being in the icu and
    a > 1 probability of being intubated.
    """
    return np.maximum(low, np.minimum(high, value))


def get_p_icu_if_recover(hr_all_age: pd.Series, hospital_parameters: 'HospitalParameters'):
    """Get probability of going to ICU given the patient recovered
    This fixes the long term average of [# used ICU beds]/[# hospitalized]
    to be expected ICU ratio.
    """
    # noinspection PyTypeChecker
    prob_death = 1 / hr_all_age
    prob_recover = 1 - prob_death

    icu_ratio = hospital_parameters.icu_ratio
    days_to_death_prob = hospital_parameters.hospital_stay_death + 1
    days_in_hosp_no_icu_prob = hospital_parameters.hospital_stay_recover + 1
    days_in_hosp_icu_prob = hospital_parameters.hospital_stay_recover_icu + 1
    days_in_icu_recover_prob = hospital_parameters.icu_stay_recover + 1

    # Fixme: For some reason, this can be negative.
    prob_icu_if_recover = (
            (icu_ratio * (days_to_death_prob * prob_death + days_in_hosp_no_icu_prob * prob_recover)
             - days_to_death_prob * prob_death)
            / ((icu_ratio * (days_in_hosp_no_icu_prob - days_in_hosp_icu_prob)
                + days_in_icu_recover_prob) * prob_recover)
    )

    return _bound(0, 1, prob_icu_if_recover)


def get_p_int_if_icu_and_recover(hr_all_age: pd.Series, hospital_parameters: 'HospitalParameters'):
    """Get the probability of intubation among recovered ICU patients.
    This fixes the long term average of [# intubated]/[# in icu] to be
    the expected intubation ratio.
    """
    # noinspection PyTypeChecker
    prob_death = 1 / hr_all_age
    prob_recover = 1 - prob_death

    prob_icu = get_p_icu_if_recover(hr_all_age, hospital_parameters)

    int_ratio = hospital_parameters.intubation_ratio
    days_to_death_prob = hospital_parameters.hospital_stay_death + 1
    days_in_hosp_icu_prob = hospital_parameters.hospital_stay_recover_icu + 1

    # Fixme: For some reason, this can be negative. This doesn't really make
    #    sense, but it's how the R code works.
    prob_int = (
            (int_ratio * (days_to_death_prob * prob_death + days_in_hosp_icu_prob * prob_icu * prob_recover)
             - days_to_death_prob * prob_death)
            / (days_in_hosp_icu_prob * prob_icu * prob_recover)
    )
    # If the probability of icu is 0, the probability of being intubated
    # condtional on ICU admission doesnt matter.
    prob_int[prob_icu == 0] = 1

    return _bound(0, 1, prob_int)


def read_correction_data(correction_path):
    df = pd.read_csv(correction_path)
    df = df.loc[(df.age_group_id == 22) & (df.sex_id == 3), ['location_id', 'date', 'value']]
    return df


def _to_census(admissions: pd.Series, length_of_stay: int) -> pd.Series:
    return (admissions
            .groupby('location_id')
            .transform(lambda x: x.rolling(length_of_stay).sum())
            .fillna(0))


def compute_hospital_usage(all_age_deaths: pd.DataFrame,
                           death_weights: pd.Series,
                           hospital_fatality_ratio: HospitalFatalityRatioData,
                           hospital_parameters: 'HospitalParameters') -> HospitalMetrics:
    all_age_deaths = all_age_deaths.set_index(['location_id', 'date'])['deaths'].sort_index()
    age_specific_deaths = (all_age_deaths * death_weights).reorder_levels(['location_id', 'age', 'date'])

    prob_icu = get_p_icu_if_recover(hospital_fatality_ratio.all_age, hospital_parameters)
    prob_no_icu = 1 - prob_icu
    prob_invasive = get_p_int_if_icu_and_recover(hospital_fatality_ratio.all_age, hospital_parameters)
    # noinspection PyTypeChecker
    # For each death, calculate number of hospital admissions of people who
    # don't die and shift back in time.
    recovered_hospital_admissions = (
        (age_specific_deaths * (hospital_fatality_ratio.age_specific - 1))
        .groupby(['location_id', 'date'])
        .sum()
        .groupby('location_id')
        .shift(-(hospital_parameters.hospital_stay_death - 1), fill_value=0)
    )
    # Split people into those who go to ICU and those who don't and count the
    # days they spend in the hospital.
    recovered_hospital_census = (
        _to_census(prob_no_icu * recovered_hospital_admissions, hospital_parameters.hospital_stay_recover)
        + _to_census(prob_icu * recovered_hospital_admissions, hospital_parameters.hospital_stay_recover_icu)
    )
    # Scale down hospitalizations to those who go to ICU and shift forward.
    recovered_icu_admissions = (
        (prob_icu * recovered_hospital_admissions)
        .groupby('location_id')
        .shift(hospital_parameters.hospital_to_icu, fill_value=0)
    )
    # Count number of days those who go to ICU spend there.
    recovered_icu_census = _to_census(recovered_icu_admissions, hospital_parameters.icu_stay_recover)
    # Ventilation usage is just scaled ICU usage.
    recovered_ventilator_census = prob_invasive * recovered_icu_census

    # Every death corresponds to a hospital admission shifted back some number
    # of days.
    dead_hospital_admissions = (
        all_age_deaths
        .groupby('location_id')
        .shift(-(hospital_parameters.hospital_stay_death - 1), fill_value=0)
    )
    # Count days from admission to get hospital census for those who die.
    dead_hospital_census = _to_census(dead_hospital_admissions, hospital_parameters.hospital_stay_death)
    # Assume people who die after entering the hospital are intubated in the
    # ICU for their full stay.
    dead_icu_admissions = dead_hospital_admissions.copy()
    dead_icu_census = dead_hospital_census.copy()
    dead_ventilator_census = dead_icu_census.copy()

    def _combine(recovered, dead):
        # Drop data after the last admission since it will be incomplete.
        return (
            (recovered + dead)
            .groupby(['location_id'])
            .apply(lambda x: x.iloc[:-(hospital_parameters.hospital_stay_death - 1)])
            .reset_index(level=0, drop=True)
        )

    return HospitalMetrics(
        hospital_admissions=_combine(recovered_hospital_admissions, dead_hospital_admissions),
        hospital_census=_combine(recovered_hospital_census, dead_hospital_census),
        icu_admissions=_combine(recovered_icu_admissions, dead_icu_admissions),
        icu_census=_combine(recovered_icu_census, dead_icu_census),
        ventilator_census=_combine(recovered_ventilator_census, dead_ventilator_census),
    )


def _compute_correction_factor(raw_log_cf: pd.Series,
                               min_date: pd.Timestamp,
                               max_date: pd.Timestamp,
                               smoothing_window: int) -> pd.Series:
    date_index = pd.date_range(min_date, max_date).rename('date')

    data_locs = raw_log_cf.reset_index().location_id.unique().tolist()
    log_cf = []
    for location_id in data_locs:
        raw_log_cf_loc = raw_log_cf.loc[location_id].reset_index()
        raw_log_cf_loc['int_date'] = raw_log_cf_loc['date'].astype(int)
        # This log space mean is a geometric mean in natural units.
        log_cf_loc = raw_log_cf_loc[['int_date', 'log_cf']].rolling(smoothing_window).mean().dropna()
        log_cf_loc['date'] = pd.to_datetime(log_cf_loc['int_date'])
        # Interpolate our moving average to fill gaps.
        log_cf_loc = log_cf_loc.set_index('date').reindex(date_index, method='nearest')
        log_cf_loc['location_id'] = location_id
        log_cf_loc = log_cf_loc.reset_index().set_index(['location_id', 'date']).log_cf
        log_cf.append(log_cf_loc)
    log_cf = pd.concat(log_cf)

    return log_cf


def _fill_missing_locations(log_cf: pd.Series, aggregation_hierarchy: pd.DataFrame) -> pd.Series:
    data_locs = log_cf.reset_index().location_id.unique().tolist()
    # Aggregate up the hierarchy using as many of the children as we can.
    # First build a map between locations and all their children.
    all_children_map = defaultdict(list)
    path_to_top_map = aggregation_hierarchy.set_index('location_id').path_to_top_parent.to_dict()
    for child_id, path_to_top_parent in path_to_top_map.items():
        for parent_id in path_to_top_parent.split(','):
            parent_id = int(parent_id)
            if parent_id != child_id:
                all_children_map[parent_id].append(child_id)

    # Take the mean in log space (geometric mean in normal space) by date of all children.
    parent_ids = aggregation_hierarchy[aggregation_hierarchy.most_detailed == 0].location_id.unique()
    for parent_id in parent_ids:
        children = all_children_map[parent_id]
        modeled_children = set(data_locs).intersection(children)
        if modeled_children:
            parent_log_cf = log_cf.loc[modeled_children].groupby(level='date').mean().reset_index()
            parent_log_cf['location_id'] = parent_id
            parent_log_cf = parent_log_cf.set_index(['location_id', 'date']).log_cf
            log_cf = log_cf.append(parent_log_cf)
            data_locs.append(parent_id)

    # Fill back in with the nearest parent
    levels = sorted(aggregation_hierarchy.level.unique().tolist())
    for level in levels[:-1]:
        parents_at_level = aggregation_hierarchy[aggregation_hierarchy.level == level].location_id.unique()
        for parent_id in parents_at_level:
            assert parent_id in data_locs
            children = aggregation_hierarchy[aggregation_hierarchy.parent_id == parent_id].location_id.unique()
            for child_id in children:
                if child_id not in data_locs:
                    child_log_cf = log_cf.loc[parent_id].reset_index()
                    child_log_cf['location_id'] = child_id
                    child_log_cf = child_log_cf.set_index(['location_id', 'date']).log_cf
                    log_cf = log_cf.append(child_log_cf)
                    data_locs.append(child_id)

    assert not set(data_locs).difference(aggregation_hierarchy.location_id)

    return log_cf


def _safe_log_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """We need to do a bunch of geometric means of ratios. This is just the
    average log difference.  This mean is poorly behaved for numbers between
    0 and 1 and so we add 1 to the numerator and denominator. This is
    approximately correct in all situations we care about and radically
    improves the behavior for small numbers."""
    # noinspection PyTypeChecker
    return (np.log(numerator + 1) - np.log(denominator + 1)).dropna().rename('log_cf')


def calculate_hospital_correction_factors(usage: 'HospitalMetrics',
                                          census_data: 'HospitalCensusData',
                                          aggregation_hierarchy: pd.DataFrame,
                                          hospital_parameters: 'HospitalParameters') -> HospitalCorrectionFactors:
    date = usage.hospital_census.reset_index().date
    min_date, max_date = date.min(), date.max()

    raw_log_hospital_cf = _safe_log_divide(census_data.hospital_census, usage.hospital_census)

    log_hospital_cf = _compute_correction_factor(
        raw_log_hospital_cf,
        min_date, max_date,
        hospital_parameters.correction_factor_smooth_window,
    )
    hospital_cf = np.exp(_fill_missing_locations(log_hospital_cf, aggregation_hierarchy))
    hospital_cf = hospital_cf.clip(lower=hospital_parameters.hospital_correction_factor_min,
                                   upper=hospital_parameters.hospital_correction_factor_max)

    modeled_icu_log_ratio = _safe_log_divide(usage.icu_census, usage.hospital_census)
    historical_icu_log_ratio = _safe_log_divide(census_data.icu_census, census_data.hospital_census)
    raw_log_icu_ratio_cf = (historical_icu_log_ratio - modeled_icu_log_ratio).dropna()

    log_icu_ratio_cf = _compute_correction_factor(
        raw_log_icu_ratio_cf,
        min_date, max_date,
        hospital_parameters.correction_factor_smooth_window,
    )
    log_icu_cf = (modeled_icu_log_ratio + log_icu_ratio_cf).dropna()
    icu_cf = np.exp(_fill_missing_locations(log_icu_cf, aggregation_hierarchy))
    icu_cf = icu_cf.clip(lower=hospital_parameters.icu_correction_factor_min,
                         upper=hospital_parameters.icu_correction_factor_max)

    modeled_ventilator_log_ratio = _safe_log_divide(usage.ventilator_census, usage.icu_census)
    historical_ventilator_log_ratio = _safe_log_divide(census_data.ventilator_census, census_data.icu_census)
    raw_log_ventilator_ratio_cf = (historical_ventilator_log_ratio - modeled_ventilator_log_ratio).dropna()

    log_ventilator_ratio_cf = _compute_correction_factor(
        raw_log_ventilator_ratio_cf,
        min_date, max_date,
        hospital_parameters.correction_factor_smooth_window,
    )
    log_ventilator_cf = (modeled_ventilator_log_ratio + log_ventilator_ratio_cf).dropna()
    ventilator_cf = np.exp(_fill_missing_locations(log_ventilator_cf, aggregation_hierarchy))
    ventilator_cf = ventilator_cf.clip(lower=hospital_parameters.intubation_correction_factor_min,
                                       upper=hospital_parameters.intubation_correction_factor_max)

    return HospitalCorrectionFactors(
        hospital_census=hospital_cf,
        icu_census=icu_cf,
        ventilator_census=ventilator_cf,
    )
