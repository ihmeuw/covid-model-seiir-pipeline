from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __OOSHoldoutJobs(NamedTuple):
    oos_regression: str
    oos_beta_scaling: str
    oos_forecast: str
    oos_join_sentinel: str
    oos_postprocess: str
    oos_diagnostics: str


OOS_HOLDOUT_JOBS = __OOSHoldoutJobs(*__OOSHoldoutJobs._fields)


class OOSRegressionTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class OOSScalingTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 26


class OOSForecastTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 15000
    default_m_mem_free = '50G'
    default_num_cores = 11


class OOSJoinSentinelTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 100
    default_m_mem_free = '1G'
    default_num_cores = 1


class OOSPostprocessingTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 11


class OOSDiagnosticsTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 5000
    default_m_mem_free = '50G'
    default_num_cores = 26


class OOSWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        OOS_HOLDOUT_JOBS.oos_regression: OOSRegressionTaskSpecification,
        OOS_HOLDOUT_JOBS.oos_beta_scaling: OOSScalingTaskSpecification,
        OOS_HOLDOUT_JOBS.oos_forecast: OOSForecastTaskSpecification,
        OOS_HOLDOUT_JOBS.oos_join_sentinel: OOSJoinSentinelTaskSpecification,
        OOS_HOLDOUT_JOBS.oos_postprocess: OOSPostprocessingTaskSpecification,
        OOS_HOLDOUT_JOBS.oos_diagnostics: OOSDiagnosticsTaskSpecification,
    }


@dataclass
class OOSHoldoutData:
    seir_forecast_version: str = field(default='best')
    seir_forecast_scenario: str = field(default='reference')
    output_root: str = field(default='')
    output_format: str = field(default='parquet')

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


@dataclass
class OOSHoldoutParameters:
    run_regression: bool = field(default=True)
    holdout_weeks: int = field(default=8)

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


class OOSHoldoutSpecification(utilities.Specification):

    def __init__(self,
                 data: OOSHoldoutData,
                 workflow: OOSWorkflowSpecification,
                 parameters: OOSHoldoutParameters):
        self._data = data
        self._workflow = workflow
        self._parameters = parameters

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': OOSHoldoutData,
            'workflow': OOSWorkflowSpecification,
            'hospital_parameters': OOSHoldoutParameters,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            sub_spec_dict = utilities.filter_to_spec_fields(
                spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**sub_spec_dict)

        return tuple(sub_specs.values())

    @property
    def data(self) -> OOSHoldoutData:
        return self._data

    @property
    def workflow(self) -> OOSWorkflowSpecification:
        return self._workflow

    @property
    def parameters(self) -> OOSHoldoutParameters:
        return self._parameters

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'parameters': self.parameters.to_dict(),
        }
        return spec
