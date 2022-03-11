from collections import defaultdict
import functools
import itertools
from typing import Callable, Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from covid_model_seiir_pipeline.lib import (
    cli_tools,
    parallel,
)
from covid_model_seiir_pipeline.pipeline.fit.data import FitDataInterface
from covid_model_seiir_pipeline.pipeline.fit.specification import FitSpecification, FIT_JOBS

logger = cli_tools.task_performance_logger


def run_beta_resampling(fit_version: str, progress_bar: bool):
    logger.info(f'Starting beta_resampling.', context='setup')
    # Build helper abstractions
    specification = FitSpecification.from_version_root(fit_version)
    data_interface = FitDataInterface.from_specification(specification)
    num_cores = specification.workflow.task_specifications[FIT_JOBS.beta_resampling].num_cores

    n_draws = data_interface.get_n_draws()
    n_total_draws = data_interface.get_n_total_draws()
    n_oversample_draws = data_interface.get_n_oversample_draws()

    rhos = data_interface.load_variant_prevalence('reference')

    window_dates = build_window_dates(rhos)
    residuals = build_residuals(window_dates, data_interface, n_total_draws, num_cores,
                                progress_bar)

    failures = []
    for measure in residuals:
        measure_success = np.abs(residuals[measure]) < 1
        std = residuals.loc[measure_success, measure].std()
        failures.append(~(np.abs(residuals[measure]) < 2 * std))
    failures = pd.concat(failures, axis=1)
    total_failures = failures[failures.all(axis=1)].reset_index(level='draw_id')['draw_id']

    failure_count = total_failures.groupby('location_id').count()
    unrecoverable = failure_count[failure_count > n_oversample_draws].index.tolist()

    replace = defaultdict(list)
    potential_substitutes = list(range(n_draws, n_total_draws))
    used_substitutes = defaultdict(list)
    for location_id, draw in total_failures.iteritems():
        if location_id in unrecoverable or draw >= n_draws:
            continue
        cant_use = used_substitutes[location_id] + total_failures[[location_id]].tolist()
        substitute_draw = [d for d in potential_substitutes if d not in cant_use][0]
        replace[draw].append((location_id, substitute_draw))

    failures = failures.reorder_levels(['draw_id', 'location_id']).sort_index()

    build_and_write_beta_finals(
        replacements=replace,
        unrecoverable=unrecoverable,
        failures=failures,
        data_interface=data_interface,
        n_draws=n_draws,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )


def build_window_dates(rhos: pd.DataFrame) -> pd.DataFrame:
    def _get_dates(variant, threshold):
        dates = (rhos
                 .loc[rhos[variant] > threshold]
                 .reset_index()
                 .groupby('location_id')
                 .date
                 .first())
        return dates

    return pd.concat([
        _get_dates("delta", 0.01).rename('delta_start'),
        _get_dates("omicron", 0.01).rename('delta_end'),
        _get_dates("omicron", 0.10).rename('omicron_start'),
    ], axis=1).dropna()


def compute_group_residual(df: pd.DataFrame) -> pd.Series:
    # df with index (location_id, date) and columns
    # (beta, infections, delta_start, delta_end, omicron_start)
    df = df.reset_index(level='location_id', drop=True)
    delta_era_data = df.loc[df['delta_start'].iloc[0]:df['delta_end'].iloc[0]]
    delta_era_mean_beta = (
        (delta_era_data['beta'] * delta_era_data['infections']).sum()
        / delta_era_data['infections'].sum()
    )
    omicron_era_data = df.loc[df['omicron_start'].iloc[0]:]
    return (
        (omicron_era_data['infections'] * (omicron_era_data['beta'] - delta_era_mean_beta)).sum()
        / omicron_era_data['infections'].sum()
    )


def build_residual(measure_draw: str, window_dates: pd.DataFrame,
                   data_interface: FitDataInterface) -> pd.Series:
    measure, draw_id = measure_draw.split('_')
    draw_id = int(draw_id)

    beta = data_interface.load_fit_beta(draw_id, measure, [f'beta_{measure}', 'round'])
    beta = beta.loc[beta['round'] == 2].drop(columns='round')[f'beta_{measure}'].rename(
        'beta')
    beta[np.abs(beta) < 1e-5] = np.nan
    infections = data_interface.load_posterior_epi_measures(draw_id, measure,
                                                            ['daily_total_infections',
                                                             'round'])
    infections = infections.loc[infections['round'] == 2].drop(columns='round')[
        'daily_total_infections'].rename('infections')
    window_dates = window_dates.reindex(beta.index, level='location_id')
    return (pd.concat([beta, infections, window_dates], axis=1)
            .dropna()
            .groupby('location_id')
            .apply(compute_group_residual)
            .rename(measure_draw))


def build_residuals(window_dates: pd.DataFrame, data_interface: FitDataInterface,
                    total_draws: int, num_cores: int, progress_bar: bool) -> pd.DataFrame:
    _runner = functools.partial(
        build_residual,
        window_dates=window_dates,
        data_interface=data_interface,
    )
    arg_list = [f'{measure}_{draw}' for measure, draw
                in itertools.product(['case', 'admission', 'death'], range(total_draws))]
    old_err_settings = np.seterr(all='ignore')
    results = parallel.run_parallel(
        runner=_runner,
        arg_list=arg_list,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    np.seterr(**old_err_settings)
    df = pd.concat(results, axis=1).sort_index().stack().reset_index()
    df.columns = ['location_id', 'measure_draw', 'value']
    df['measure'], df['draw_id'] = df.measure_draw.str.split('_').str
    df['draw_id'] = df['draw_id'].astype(int)
    df = df.set_index(['location_id', 'draw_id', 'measure']).value.unstack().sort_index()
    df.columns.name = None
    return df


def make_error_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
    for i, measure in enumerate(df.columns):
        s = df[measure]
        ax[i].axvline(s.mean(), linewidth=3, color='k')
        s = s[np.abs(s) < 1]
        ax[i].axvline(s.mean(), linewidth=3, color='k', linestyle='--')
        s.plot.hist(bins=100, ax=ax[i])
        ax[i].axvline(0, linewidth=3, color='red')
        ax[i].set_title(measure.capitalize(), fontsize=18)
    plt.show()


def load_data_subset(draw_id: int,
                     loader: Callable[[int, str, List[str]], pd.DataFrame],
                     columns: List[str],
                     draw_replace: Dict[int, List[int]]) -> pd.DataFrame:
    data = []
    for measure in ['case', 'death', 'admission']:
        df = loader(draw_id, measure, [c.format(measure=measure) for c in columns] + ['round'])
        df = df.loc[df['round'] == 2].drop(columns='round')
        df = df.rename(columns=lambda x: x if measure in x else f'{x}_{measure}')
        data.append(df)
    data = pd.concat(data, axis=1)
    if draw_id in draw_replace:
        keep_idx = draw_replace[draw_id]
    else:
        drop = [loc_id for loc_list in draw_replace.values() for loc_id in loc_list]
        keep_idx = [loc_id for loc_id in data.reset_index().location_id.unique() if loc_id not in drop]
    return data.loc[keep_idx]

def add_location_id(df, location_id):
    df = df.reset_index()
    df['location_id'] = location_id
    df = df.set_index(['location_id', 'date']).sort_index()
    return df


def build_and_write_beta_final(draw_id: int,
                               replacements: Dict[int, List[Tuple[int, int]]],
                               unrecoverable: List[int],
                               failures: pd.DataFrame,
                               data_interface: FitDataInterface) -> None:
    draw_replace = defaultdict(list)
    for location_id, replace_list in replacements.items():
        for d, s in replace_list:
            if d == draw_id:
                draw_replace[s].append(location_id)
    draws_to_load = [draw_id] + list(draw_replace)
    betas = []
    infections = []
    for d in draws_to_load:
        betas.append(load_data_subset(
            draw_id=d,
            loader=data_interface.load_fit_beta,
            columns=['beta_{measure}'],
            draw_replace=draw_replace,
        ))
        infections.append(load_data_subset(
            draw_id=d,
            loader=data_interface.load_posterior_epi_measures,
            columns=['daily_total_infections', 'daily_naive_infections'],
            draw_replace=draw_replace,
        ))
    betas = pd.concat(betas).sort_index()
    infections = pd.concat(infections).sort_index()

    draw_failures = failures.loc[draw_id]
    for draw, locs in draw_replace.items():
        replace_failures = failures.loc[draw].loc[locs]
        draw_failures = draw_failures.drop(replace_failures.index).append(replace_failures).sort_index()

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
    data_interface.save_fit_beta(final_betas, draw_id, measure_version='final')
    data_interface.save_posterior_epi_measures(final_infections, draw_id, measure_version='resampled')


def build_and_write_beta_finals(replacements: Dict[int, List[Tuple[int, int]]],
                                unrecoverable: List[int],
                                failures: pd.DataFrame,
                                data_interface: FitDataInterface,
                                n_draws: int,
                                num_cores: int,
                                progress_bar: bool):
    _runner = functools.partial(
        build_and_write_beta_final,
        replacements=replacements,
        unrecoverable=unrecoverable,
        failures=failures,
        data_interface=data_interface,
    )
    arg_list = list(range(n_draws))
    parallel.run_parallel(
        runner=_runner,
        arg_list=arg_list,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )


@click.command()
@cli_tools.with_task_fit_version
@cli_tools.add_verbose_and_with_debugger
@cli_tools.with_progress_bar
def beta_resampling(fit_version: str,
                    progress_bar: bool, verbose: int, with_debugger: bool):
    cli_tools.configure_logging_to_terminal(verbose)
    run = cli_tools.handle_exceptions(run_beta_resampling, logger, with_debugger)
    run(fit_version=fit_version,
        progress_bar=progress_bar)
