from covid_model_seiir_pipeline.pipeline.postprocessing.specification import (
    POSTPROCESSING_JOBS,
    PostprocessingSpecification,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.data import PostprocessingDataInterface
from covid_model_seiir_pipeline.pipeline.postprocessing.task import (
    resample_map,
    postprocess,
)
from covid_model_seiir_pipeline.pipeline.postprocessing.main import (
    postprocess,
    do_postprocessing,
)

SPECIFICATION = PostprocessingSpecification
COMMAND = postprocess
APPLICATION_MAIN = do_postprocessing
TASKS = {
    POSTPROCESSING_JOBS.resample: resample_map,
    POSTPROCESSING_JOBS.postprocess: postprocess,
}
