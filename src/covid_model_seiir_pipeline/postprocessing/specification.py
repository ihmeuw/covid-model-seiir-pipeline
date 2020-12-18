from dataclasses import dataclass, field
from typing import Dict, NamedTuple

from covid_model_seiir_pipeline.workflow_tools.specification import (
    TaskSpecification,
    WorkflowSpecification,
)


class __PostprocessingJobs(NamedTuple):
    postprocess: str = 'postprocess'


POSTPROCESSING_JOBS = __PostprocessingJobs()


class PostprocessingTaskSpecification(TaskSpecification):
    """Specification of execution parameters for postprocessing tasks."""
    default_max_runtime_seconds = 15000
    default_m_mem_free = '150G'
    default_num_cores = 26


class PostprocessingWorkflowSpecification(WorkflowSpecification):
    """Specification of execution parameters for forecasting workflows."""

    tasks = {
        POSTPROCESSING_JOBS.postprocess: PostprocessingTaskSpecification,
    }


@dataclass
class PostprocessingData:
    """Specifies the inputs and outputs for postprocessing."""
    forecast_version: str = field(default='best')
    include_scenarios: list = field(default_factory=lambda: ['worse', 'reference', 'best_masks'])
    output_root: str = field(default='')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return asdict(self)
