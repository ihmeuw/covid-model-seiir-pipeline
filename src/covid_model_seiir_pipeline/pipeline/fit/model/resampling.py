
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.pipeline.fit.data import (
    FitDataInterface,
)


def load_and_resample_beta_and_infections(draw_id: int,
                                          data_interface: FitDataInterface):
    failures = data_interface.load_fit_failures()
    resampling_map = data_interface.load_draw_resampling_map()
    # List of totally failed locations
    unrecoverable = resampling_map['unrecoverable']
    # List of (location_id, substitute_draw)
    replace_list = resampling_map['replacements_by_draw'][draw_id]
    replace_map = defaultdict(list)
    for location_id, substitute_draw in replace_list:
        replace_map[substitute_draw].append(location_id)

    betas = []
    infections = []
    for d in [draw_id] + list(replace_map):
        betas.append(load_data_subset(
            draw_id=d,
            loader=data_interface.load_fit_beta,
            columns=['beta_{measure}'],
            replace_map=replace_map,
        ))
        infections.append(load_data_subset(
            draw_id=d,
            loader=data_interface.load_posterior_epi_measures,
            columns=['daily_total_infections', 'daily_naive_infections'],
            replace_map=replace_map,
        ))
    betas = pd.concat(betas).sort_index()
    infections = pd.concat(infections).sort_index()

    draw_failures = failures.loc[draw_id]
    for draw, locs in replace_map.items():
        replace_failures = failures.loc[draw].loc[locs]
        draw_failures = (draw_failures
                         .drop(replace_failures.index)
                         .append(replace_failures)
                         .sort_index())

    final_betas = []
    final_infections = []
    for location_id in betas.reset_index().location_id.unique():
        loc_beta = betas.loc[location_id]
        loc_infections = infections.loc[location_id]
        if location_id in unrecoverable:
            # We can't do anything, this totally failed.
            loc_beta = pd.concat([loc_beta, pd.Series(np.nan, name='beta', index=loc_beta.index)], axis=1)
            final_betas.append(add_location_id(loc_beta, location_id))
            final_infections.append(add_location_id(loc_infections, location_id))
            continue

        # Some locations like China, never had delta, so there's no error.
        if location_id in draw_failures.index:
            loc_failures = draw_failures.loc[location_id]
            for measure, measure_failed in loc_failures.iteritems():
                if measure_failed:
                    loc_beta[f'beta_{measure}'] = np.nan
                    loc_infections[f'daily_total_infections_{measure}'] = np.nan
                    loc_infections[f'daily_naive_infections_{measure}'] = np.nan

        loc_beta_mean = loc_beta.mean(axis=1).rename('beta')
        x = loc_beta_mean.dropna().reset_index()
        # get a date in the middle of the series to use in the intercept shift
        # so we avoid all the nonsense at the beginning.
        index_date, level = x.iloc[len(x) // 2]
        loc_beta_diff_mean = loc_beta.diff().mean(axis=1).cumsum().rename('beta')
        loc_beta_diff_mean += level - loc_beta_diff_mean.loc[index_date]
        loc_beta_diff_mean = pd.concat([loc_beta_mean.loc[:index_date],
                                        loc_beta_diff_mean.loc[
                                        index_date + pd.Timedelta(days=1):]])
        loc_beta_diff_mean = add_location_id(loc_beta_diff_mean, location_id)['beta']
        loc_beta = loc_beta.reindex(loc_beta_diff_mean.index, level='date')
        final_betas.append(pd.concat([loc_beta, loc_beta_diff_mean], axis=1))
        final_infections.append(add_location_id(loc_infections, location_id))
    final_betas = pd.concat(final_betas).sort_index()
    final_infections = pd.concat(final_infections).sort_index()
    return final_betas, final_infections


def load_data_subset(draw_id: int,
                     loader: Callable[[int, str, List[str]], pd.DataFrame],
                     columns: List[str],
                     replace_map: Dict[int, List[int]]) -> pd.DataFrame:
    data = []
    for measure in ['case', 'death', 'admission']:
        df = loader(draw_id, measure, [c.format(measure=measure) for c in columns] + ['round'])
        df = df.loc[df['round'] == 2].drop(columns='round')
        df = df.rename(columns=lambda x: x if measure in x else f'{x}_{measure}')
        data.append(df)
    data = pd.concat(data, axis=1)
    if draw_id in replace_map:
        keep_idx = replace_map[draw_id]
    else:
        drop = [loc_id for loc_list in replace_map.values() for loc_id in loc_list]
        keep_idx = [loc_id for loc_id in data.reset_index().location_id.unique() if loc_id not in drop]
    return data.loc[keep_idx]


def add_location_id(df, location_id):
    df = df.reset_index()
    df['location_id'] = location_id
    df = df.set_index(['location_id', 'date']).sort_index()
    return df

