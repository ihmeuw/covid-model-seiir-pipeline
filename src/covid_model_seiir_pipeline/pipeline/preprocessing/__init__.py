from covid_model_seiir_pipeline.pipeline.preprocessing.specification import (
    PREPROCESSING_JOBS,
    PreprocessingSpecification,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.data import (
    PreprocessingDataInterface,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.task import (
    preprocess_measure,
    preprocess_vaccine,
)
from covid_model_seiir_pipeline.pipeline.preprocessing.main import (
    do_preprocessing,
    preprocess,
)

SPECIFICATION = PreprocessingSpecification
COMMAND = preprocess
APPLICATION_MAIN = do_preprocessing
TASKS = {
    PREPROCESSING_JOBS.preprocess_measure: preprocess_measure,
    PREPROCESSING_JOBS.preprocess_vaccine: preprocess_vaccine,
}
