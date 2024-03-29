from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from regmod.data import Data
from regmod.variable import Variable, SplineVariable
from regmod.utils import SplineSpecs
from regmod.models import GaussianModel

from covid_model_seiir_pipeline.lib import (
    parallel,
)


def build_composite_betas(betas: pd.DataFrame,
                          infections: pd.DataFrame,
                          alpha: pd.Series,
                          num_cores: int,
                          progress_bar: bool,
                          location_ids: List[int] = None) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if location_ids is None:
        location_ids = betas.reset_index()['location_id'].unique()
    
    arg_list = []
    for location_id in location_ids:
        arg_list.append((
            infections.loc[[location_id]],
            betas.loc[[location_id]],
            alpha.loc[location_id],
        ))

    results = parallel.run_parallel(
        make_composite_beta,
        arg_list=arg_list,
        num_cores=num_cores,
        progress_bar=progress_bar,
    )
    beta, infections, infectious = [pd.concat(dfs) for dfs in zip(*results)]    
    return beta, infections, infectious


def make_composite_beta(args: Tuple):
    infections, betas, alpha = args
    infectious = []
    for measure in ['case', 'death', 'admission']:
        infectious.append(
            ((infections.loc[:, f'infection_{measure}'] / betas.loc[:, f'beta_{measure}']) ** (1 / alpha))
            .rename(f'I_{measure}')
        )
    infectious = pd.concat(infectious, axis=1)

    composite_infectious = combination_spline(infectious)
    composite_infections = combination_spline(infections.loc[composite_infectious.index])

    composite_beta = (composite_infections / (composite_infectious ** alpha)).rename('beta_all_infection')

    return composite_beta, composite_infections.rename('infections'), composite_infectious.rename('infectious')


def combination_spline(data: pd.DataFrame):
    data_idx = data.notnull().any(axis=1)
    dates = data.loc[data_idx].reset_index()['date'][1:].reset_index(drop=True)
    n_days = (dates.max() - dates.min()).days

    # determine spline specs
    if n_days > 90:
        # start anchor (using mean up to last 30 days)
        k_start = np.array([0])

        # tighter over last 45 days
        k_end = np.linspace(n_days - 45, n_days, 10)

        # every 45 days during middle interval
        k_middle = np.linspace(k_start.max(), k_end.min(),
                               int((k_end.min() - k_start.max()) / 45) + 1)[1:-1]

        # stitch together
        knots = np.hstack([k_start, k_middle, k_end]) / n_days
        spline_specs = dict(
            knots=knots,
            l_linear=True,
            r_linear=True,
        )
    else:
        # weekly for short time series
        knots = np.linspace(0, n_days, int(n_days / 10)) / n_days
        if knots.size > 4:
            spline_specs = dict(
                knots=knots,
                l_linear=True,
                r_linear=True,
            )
        else:
            spline_specs = dict(
                knots=knots,
            )
    spline_specs.update(
        dict(knots_type='rel_domain',
             include_first_basis=False,
             degree=3,)
    )

    # prepare model inputs
    delta_log_data = np.log(data.loc[data_idx].clip(1, np.inf)).diff()
    leading_null = data.loc[data_idx].ffill().isnull() & data.loc[data_idx].bfill().notnull()
    data = data.loc[data_idx].where(~leading_null, other=0)

    # prepare prediction dataframe
    pred_data_template = data.loc[data_idx].reset_index().loc[:, ['location_id', 'date']].drop_duplicates()

    # run standard and delta models
    pred_data = data.mean(axis=1) # fit_spline(data, pred_data_template.copy(), spline_specs,)
    pred_delta_log_data = fit_spline(delta_log_data, pred_data_template.copy(), spline_specs,)

    # splice average and delta ln() if we are combining measures, otherwise just take the single measure as-is
    if data.notnull().any(axis=0).sum() > 1:
        pred_data = pd.concat([
            pred_data[:-30],
            np.exp(np.log(pred_data.iloc[-31] + 0.03) + pred_delta_log_data.loc[data_idx][-30:].cumsum())
        ])
    pred_data = pred_data.rename('pred').clip(1e-4, np.inf)

    return pred_data


def fit_spline(data: pd.DataFrame,
               pred_data: pd.DataFrame,
               spline_specs: Dict,):
    # format data
    data = data.stack().reset_index().dropna()
    data.columns = ['location_id', 'date', 'measure', 'data']
    data['intercept'] = 1
    t0 = data['date'].min()
    data['t'] = (data['date'] - t0).dt.days

    # create model data structures, fit model, and predict out
    model_data = Data(col_obs='data', col_covs=['intercept', 't'], df=data)
    model_variables = [Variable(name='intercept',),
                       SplineVariable(name='t',
                                      spline_specs=SplineSpecs(**spline_specs),)]
    model = GaussianModel(data=model_data, param_specs={'mu': {'variables': model_variables}})
    model.fit()
    pred_data['intercept'] = 1
    pred_data['t'] = (pred_data['date'] - t0).dt.days
    pred_data = (model.predict(pred_data)
                 .rename(columns={'mu': 'pred_data'}))

    # make cumulative again, add back residual, and exponentiate
    pred_data = (pred_data
                 .set_index(['location_id', 'date'])
                 .loc[:, 'pred_data'])

    return pred_data
