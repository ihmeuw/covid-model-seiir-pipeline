from typing import Dict, List, Iterable, Optional, TYPE_CHECKING

import pandas as pd
import numpy as np

from covid_model_seiir_pipeline.pipeline.regression.model import reslime

if TYPE_CHECKING:
    # The model subpackage is a library for the pipeline stage and shouldn't
    # explicitly depend on things outside the subpackage.
    from covid_model_seiir_pipeline.pipeline.regression.specification import CovariateSpecification


def prep_regression_weights(
    infections: pd.Series,
    rhos: pd.DataFrame,
    population: pd.Series,
    hierarchy: pd.DataFrame,
    weighting_scheme: str
) -> pd.Series:
    infection_weights = _location_specific_normalization(
        infections=infections,
        max_infections=infections.groupby('location_id').max(),
    )
    ancestral_infection_weights = _location_specific_normalization(
        infections=infections,
        max_infections=_max_infections_by_variant(infections, rhos, ['ancestral'])
    )

    infection_rate = infections / population.reindex(infections.index, level='location_id')
    infection_rate_weights = ((infection_rate / infection_rate.quantile(.75))
                              .fillna(0.)
                              .rename('weight')
                              .clip(0.0, 1.0))

    weights = {
        '': pd.Series(1., index=infections.index, name='weight'),
        'infection': infection_weights,
        'infection_rate': infection_rate_weights,
        'threshold_one': _apply_threshold(infection_weights, 0.01),
        'infection_rate_threshold_one': _apply_threshold(infection_rate_weights, 0.01),
        'threshold_five': _apply_threshold(infection_weights, 0.05),
        'infection_rate_threshold_five': _apply_threshold(infection_rate_weights, 0.05),
        'ancestral_threshold_one': _apply_threshold(ancestral_infection_weights, 0.01),
        'ancestral_threshold_five': _apply_threshold(ancestral_infection_weights, 0.05),
    }
    weights = weights[weighting_scheme]

    # don't allow China or Australasia to impact fitting
    def _child_locations(location_id):
        most_detailed = hierarchy['most_detailed'] == 1
        is_child = (hierarchy['path_to_top_parent']
                    .apply(lambda x: str(location_id) in x.split(',')))
        return hierarchy.loc[most_detailed & is_child, 'location_id'].to_list()

    drop_from_regression = [
        *_child_locations(6),  # China subnationals
        71,  # Australia
        72,  # New Zealand
    ]
    modeled_locations = infections.index.get_level_values('location_id')
    drop_from_regression = [l for l in drop_from_regression if l in modeled_locations]

    # Massively downweight, but still allow for a random intercept.
    weights.loc[drop_from_regression] = 0.001
    return weights


def _max_infections_by_variant(infections: pd.Series, rhos: pd.DataFrame, keep_variants: List[str] = None) -> pd.Series:
    keep_variants = keep_variants if keep_variants is not None else list(rhos)
    last_date = (rhos[rhos[keep_variants].sum(axis=1) >=0.99]
                 .reset_index()
                 .groupby('location_id')
                 .date
                 .max()
                 .rename('last_date')
                 .reindex(infections.index, level='location_id'))
    filtered_infections = pd.concat([infections, last_date], axis=1).reset_index()
    max_infections = (filtered_infections.loc[filtered_infections.date <= filtered_infections.last_date]
                      .groupby('location_id')
                      .daily_total_infections
                      .max()
                      .rename('max_infections'))
    return max_infections


def _location_specific_normalization(infections: pd.Series, max_infections: pd.Series) -> pd.Series:
    max_infections = max_infections.reindex(infections.index, level='location_id').rename('max_infections')
    data = pd.concat([infections, max_infections], axis=1)
    data = (data
            .groupby('location_id')
            .apply(lambda x: x.daily_total_infections / x.max_infections.max())
            .reset_index(level=0, drop=True)
            .fillna(0.)
            .rename('weight')
            .clip(0.0, 1.0))
    return data


def _apply_threshold(data: pd.Series, threshold: float):
    data = data.copy()
    data[data < threshold] = 0.
    data[data >= threshold] = 1.
    return data


def run_beta_regression(beta_fit: pd.Series,
                        regression_weights: pd.Series,
                        covariates: pd.DataFrame,
                        covariate_specs: Iterable['CovariateSpecification'],
                        gaussian_priors: Dict[str, pd.DataFrame],
                        prior_coefficients: Optional[pd.DataFrame],
                        hierarchy: pd.DataFrame) -> pd.DataFrame:
    regression_inputs = prep_regression_inputs(
        beta_fit,
        regression_weights,
        covariates,
        hierarchy
    )
    predictor_set, fixed_coefficients = build_predictors(
        regression_inputs,
        covariate_specs,
        gaussian_priors,
        prior_coefficients,
    )
    mr_data = reslime.MRData(
        data=regression_inputs.reset_index(),
        response_column='ln_beta',
        weight_column='weight',
        predictors=[p.name for p in predictor_set],
        group_columns=['super_region_id', 'region_id', 'location_id'],
    )
    mr_model = reslime.MRModel(mr_data, predictor_set)
    coefficients = mr_model.fit_model().reset_index(level=['super_region_id', 'region_id'], drop=True)
    coefficients = pd.concat([coefficients, *fixed_coefficients], axis=1)
    return coefficients


def prep_regression_inputs(beta_fit: pd.Series,
                           regression_weights: pd.Series,
                           covariates: pd.DataFrame,
                           hierarchy: pd.DataFrame):
    regression_inputs = pd.concat([beta_fit, regression_weights], axis=1)
    regression_inputs = pd.merge(regression_inputs.dropna(), covariates, on=beta_fit.index.names)
    group_cols = ['super_region_id', 'region_id', 'location_id']
    regression_inputs = (regression_inputs
                         .merge(hierarchy[group_cols], on='location_id')
                         .reset_index()
                         .set_index(group_cols)
                         .sort_index())
    regression_inputs['intercept'] = 1.0
    regression_inputs['ln_beta'] = np.log(regression_inputs['beta'])
    return regression_inputs


def build_predictors(regression_inputs: pd.DataFrame,
                     covariate_specs: Iterable['CovariateSpecification'],
                     gaussian_priors: Dict[str, pd.DataFrame],
                     prior_coefficients: Optional[pd.DataFrame]):
    location_ids = pd.Index(regression_inputs.reset_index()['location_id'].unique(), name='location_id')
    predictors = []
    fixed_coefficients = []
    for covariate in covariate_specs:
        if np.all(regression_inputs[covariate.name] == 0):
            fixed_coefficients.append(
                pd.Series(
                    0.0,
                    name=covariate.name,
                    index=location_ids,
                )
            )
        elif prior_coefficients is not None and not covariate.group_level and covariate.name in prior_coefficients:
            coefficient_val = (
                prior_coefficients
                .reset_index()
                .set_index('location_id')[covariate.name]
                .drop_duplicates()
            )
            assert len(coefficient_val.index) == 1
            coefficient_val = pd.Series(coefficient_val.iloc[0], name=covariate.name, index=location_ids)
            fixed_coefficients.append(coefficient_val)
            coefficient_val = coefficient_val.reindex(regression_inputs.index, level='location_id')
            regression_inputs['ln_beta'] -= coefficient_val * regression_inputs[covariate.name]
        else:
            predictor = reslime.PredictorModel.from_specification(covariate)
            if predictor.original_prior == 'data':
                predictor.original_prior = gaussian_priors[covariate.name]
            predictors.append(predictor)
    predictor_set = reslime.PredictorModelSet(predictors)
    return predictor_set, fixed_coefficients

