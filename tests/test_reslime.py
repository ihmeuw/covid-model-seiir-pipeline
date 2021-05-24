"""Integration tests against the old version of slime.

These are to be deleted along with slime itself as soon as we can swap to the
new model.

"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.pipeline.regression import (
    RegressionSpecification,
    model,
)
from covid_model_seiir_pipeline.pipeline.regression.model import (
    slime,
    reslime,
)


@pytest.fixture
def fixture_data_dir():
    pkg_init = Path(covid_model_seiir_pipeline.__file__)
    repo_dir = (
        pkg_init
        .parent  # pkg dir
        .parent  # src
        .parent  # repo dir
    )
    return repo_dir / 'tests' / 'fixture_data'


@pytest.fixture
def beta(fixture_data_dir):
    df = pd.read_parquet(fixture_data_dir / 'beta.parquet')
    # filter to US states for speed.
    df = df.loc[list(set(list(range(523, 574))).difference([570])), 'beta']
    return df


@pytest.fixture
def covariates(fixture_data_dir):
    df = pd.read_parquet(fixture_data_dir / 'covariates.parquet')
    return df


@pytest.fixture
def regression_spec(fixture_data_dir):
    return RegressionSpecification.from_path(fixture_data_dir / 'regression_specification.yaml')


@pytest.fixture
def all_data(beta, covariates):
    all_data = (pd.merge(beta.dropna(), covariates, on=beta.index.names)
                .sort_index()
                .reset_index())
    all_data['ln_beta'] = np.log(all_data['beta'])
    return all_data



@pytest.fixture
def old_mr_data(all_data, covariates):
    mr_data = slime.MRData(
        all_data,
        col_group='location_id',
        col_obs='ln_beta',
        col_covs=covariates.columns.tolist(),
    )
    return mr_data


@pytest.fixture
def new_mr_data(all_data, covariates):
    mr_data = reslime.MRData(
        all_data,
        response_column='ln_beta',
        predictors=covariates.columns.tolist(),
        group_columns=['location_id'],
    )
    return mr_data


@pytest.fixture
def old_model(beta, covariates, regression_spec):
    regression_inputs = model.prep_regression_inputs(
        beta,
        covariates,
    )

    regressor = model.build_regressor(regression_spec.covariates.values(), prior_coefficients=None)

    coefficients = regressor.fit(
        regression_inputs,
        regression_spec.regression_parameters.sequential_refit,
    )
    return regression_inputs, regressor, coefficients


@pytest.fixture
def old_covmodel_set(old_model):
    _, regressor, _ = old_model
    return regressor.covmodel_set


def test_manual_covariate_construction(old_covmodel_set, old_mr_data, regression_spec):
    cov_model_set = slime.CovModelSet([
        slime.CovModel.from_specification(covariate)
        for covariate in regression_spec.covariates.values()
    ])
    cov_model_set.attach_data(old_mr_data)

    for cov_model in old_covmodel_set.cov_models:
        matching_model = [c for c in cov_model_set.cov_models if c.col_cov == cov_model.col_cov].pop()
        assert matching_model == cov_model


def test_new_covariate_model(old_mr_data, new_mr_data, regression_spec):
    mobility_spec = regression_spec.covariates['mobility']
    old_cov = slime.CovModel(
        'mobility',
        use_re=True,
        bounds=mobility_spec.bounds,
        gprior=mobility_spec.gprior,
        re_var=np.inf,
    )
    old_cov.attach_data(old_mr_data)

    mobility = new_mr_data.data.set_index('location_id')['mobility']

    new_cov = reslime.PredictorModel(
        'mobility',
        group_level='location_id',
        bounds=mobility_spec.bounds,
        gaussian_prior_params=mobility_spec.gprior,
    )
    new_cov.attach_data(new_mr_data)

    assert new_cov.var_size == old_cov.var_size
    coef = np.random.random(new_cov.var_size)
    residual = 2*np.random.random(len(mobility)) - 1
    obs_se = np.random.random(len(mobility))
    assert np.all(new_cov.predict(coef) == old_cov.predict(coef))
    assert np.all(new_cov.prior_objective(coef) == old_cov.prior_objective(coef))
    assert np.all(new_cov.gradient(coef, residual, obs_se) == old_cov.gradient(coef, residual, obs_se))


def test_new_covariate_model_set(old_mr_data, new_mr_data, regression_spec):
    old_cov_model_set = slime.CovModelSet([
        slime.CovModel(
            covariate,
            use_re=covariate_spec.use_re,
            bounds=covariate_spec.bounds,
            gprior=covariate_spec.gprior,
            re_var=np.inf,
        )
        for covariate, covariate_spec in regression_spec.covariates.items()
    ])
    old_cov_model_set.attach_data(old_mr_data)

    new_cov_model_set = reslime.PredictorModelSet([
        reslime.PredictorModel(
            covariate,
            group_level='location_id' if covariate_spec.use_re else reslime.NO_GROUP,
            bounds=covariate_spec.bounds,
            gaussian_prior_params=covariate_spec.gprior,
        )
        for covariate, covariate_spec in regression_spec.covariates.items()
    ])
    new_cov_model_set.attach_data(new_mr_data)

    assert new_cov_model_set.var_size == old_cov_model_set.var_size
    coef = np.random.random(new_cov_model_set.var_size)
    residual = 2*np.random.random(len(new_mr_data.data)) - 1
    obs_se = np.random.random(len(new_mr_data.data))
    assert np.all(new_cov_model_set.predict(coef) == old_cov_model_set.predict(coef))
    assert np.all(new_cov_model_set.prior_objective(coef) == old_cov_model_set.prior_objective(coef))
    assert np.all(new_cov_model_set.gradient(coef, residual, obs_se)
                  == old_cov_model_set.gradient(coef, residual, obs_se))


def test_new_mr_model(old_mr_data, new_mr_data, regression_spec):
    old_cov_model_set = slime.CovModelSet([
        slime.CovModel(
            covariate,
            use_re=covariate_spec.use_re,
            bounds=covariate_spec.bounds,
            gprior=covariate_spec.gprior,
            re_var=np.inf,
        )
        for covariate, covariate_spec in regression_spec.covariates.items()
    ])
    old_mr_model = slime.MRModel(old_mr_data, old_cov_model_set)
    old_mr_model.fit_model()
    old_coef = pd.DataFrame.from_dict(old_mr_model.result, orient='index').reset_index()
    old_coef.columns = ['location_id'] + list(regression_spec.covariates)
    old_coef = old_coef.set_index('location_id')

    new_cov_model_set = reslime.PredictorModelSet([
        reslime.PredictorModel(
            covariate,
            group_level='location_id' if covariate_spec.use_re else reslime.NO_GROUP,
            bounds=covariate_spec.bounds,
            gaussian_prior_params=covariate_spec.gprior,
        )
        for covariate, covariate_spec in regression_spec.covariates.items()
    ])
    new_mr_model = reslime.MRModel(new_mr_data, new_cov_model_set)
    new_coef = new_mr_model.fit_model()

    assert old_coef.equals(new_coef)


def test_multilevel_regression(beta, covariates, regression_spec):
    location_ids = beta.reset_index().location_id.unique()
    group_ids = []
    for i, lg in enumerate(np.split(location_ids, 5)):
        group_ids.extend([i]*len(lg))
    group_s = pd.Series(group_ids, index=location_ids, name='region_id')
    new_beta = pd.concat([beta, group_s.reindex(beta.index, level='location_id')], axis=1)

    all_data = (pd.merge(new_beta.dropna(), covariates, on=['location_id', 'date'])
                .sort_index()
                .reset_index())
    all_data['ln_beta'] = np.log(all_data['beta'])

    mr_data = reslime.MRData(
        all_data,
        response_column='ln_beta',
        predictors=covariates.columns.tolist(),
        group_columns=['region_id', 'location_id'],
    )

    group_levels = {'mobility': 'location_id', 'intercept': 'location_id', 'mask_use': 'region_id'}
    new_cov_model_set = reslime.PredictorModelSet([
        reslime.PredictorModel(
            covariate,
            group_level=group_levels.get(covariate, reslime.NO_GROUP),
            bounds=covariate_spec.bounds,
            gaussian_prior_params=covariate_spec.gprior,
        )
        for covariate, covariate_spec in regression_spec.covariates.items()
    ])
    new_mr_model = reslime.MRModel(mr_data, new_cov_model_set)
    new_coef = new_mr_model.fit_model()

    for predictor in new_coef:
        if predictor in group_levels:
            expected_count = len(all_data[group_levels[predictor]].unique())
        else:
            expected_count = 1
        assert len(new_coef[predictor].unique()) == expected_count
