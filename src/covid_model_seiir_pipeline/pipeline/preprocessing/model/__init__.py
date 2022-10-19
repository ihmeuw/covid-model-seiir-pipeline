from covid_model_seiir_pipeline.pipeline.preprocessing.model.epi_data import (
    preprocess_epi_data,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model.covariate_data import (
    preprocess_mask_use,
    preprocess_prop_65plus,
    preprocess_mobility,
    preprocess_mandates,
    preprocess_pneumonia,
    preprocess_population_density,
    preprocess_testing_data,
    preprocess_variant_prevalence,
    preprocess_gbd_covariate,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model.sensitivity import (
    preprocess_sensitivity,
)

_GBD_COVARIATES = [
    'air_pollution_pm_2_5',
    'ckd',
    'cvd',
    'uhc',
    'haqi',
    'mean_age_standardized_sbp_mmhg_above_age_25',
    'obesity',
    'smoking_prevalence',
    'cancer',
    'copd',
    'diabetes',
    'lri_mortality',
    'mean_bmi_above_age_20',
    'proportion_under_100m',
]
COVARIATES = {
    'mask_use': preprocess_mask_use,
    'prop_65plus': preprocess_prop_65plus,
    'mobility': preprocess_mobility,
    'mandates': preprocess_mandates,
    'pneumonia': preprocess_pneumonia,
    'proportion_over_2_5k': preprocess_population_density,
    'testing': preprocess_testing_data,
    **{covariate: preprocess_gbd_covariate(covariate) for covariate in _GBD_COVARIATES},
}

MEASURES = {
    'epi_data': preprocess_epi_data,
    'variant_prevalence': preprocess_variant_prevalence,
    'sensitivity': preprocess_sensitivity,
    **COVARIATES,
}
