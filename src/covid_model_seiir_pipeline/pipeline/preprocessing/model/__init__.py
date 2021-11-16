from covid_model_seiir_pipeline.pipeline.preprocessing.model.epi_data import (
    preprocess_epi_data,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.model.covariate_data import (
    preprocess_mask_use,
    preprocess_mobility,
    preprocess_pneumonia,
    preprocess_population_density,
    preprocess_testing_data,
    preprocess_variant_prevalence,
)

MEASURES = {
    'epi_data': preprocess_epi_data,
    'mask_use': preprocess_mask_use,
    'mobility': preprocess_mobility,
    'pneumonia': preprocess_pneumonia,
    'population_density': preprocess_population_density,
    'testing': preprocess_testing_data,
    'variant_prevalence': preprocess_variant_prevalence,
}
