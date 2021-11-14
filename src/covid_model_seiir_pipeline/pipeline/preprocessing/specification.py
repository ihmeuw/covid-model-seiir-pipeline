from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Tuple

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __PreprocessingJobs(NamedTuple):
    smooth_epi_measures: str
    compute_waning: str
    covariate_prep: str
    preprocess_validation: str
    preprocess_diagnostics: str


PREPROCESSING_JOBS = __PreprocessingJobs(*__PreprocessingJobs._fields)


class SmoothEpiMeasuresTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class ComputeWaningTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class CovariatePrepTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class PreprocessingValidationTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class PreprocessingDiagnosticsTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 3000
    default_m_mem_free = '20G'
    default_num_cores = 5


class PreprocessingWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        PREPROCESSING_JOBS.smooth_epi_measures: SmoothEpiMeasuresTaskSpecification,
        PREPROCESSING_JOBS.compute_waning: ComputeWaningTaskSpecification,
        PREPROCESSING_JOBS.covariate_prep: CovariatePrepTaskSpecification,
        PREPROCESSING_JOBS.preprocess_validation: PreprocessingValidationTaskSpecification,
        PREPROCESSING_JOBS.preprocess_diagnostics: PreprocessingDiagnosticsTaskSpecification,
    }


@dataclass
class PreprocessingData:
    location_set_version_id: int = field(default=0)
    location_set_file: str = field(default='')

    model_inputs_version: str = field(default='best')
    age_specific_rates_version: str = field(default='best')
    mortality_scalars_version: str = field(default='best')
    mask_use_version: str = field(default='best')
    mobility_version: str = field(default='best')
    pneumonia_version: str = field(default='best')
    population_density_version: str = field(default='best')
    testing_version: str = field(default='best')
    vaccine_coverage_version: str = field(default='best')
    variant_prevalence_version: str = field(default='best')

    output_root: str = field(default='')
    output_format: str = field(default='parquet')
    n_draws: int = field(default=100)

    run_counties: bool = field(init=False)
    drop_locations: list = field(default_factory=list)

    def __post_init__(self):
        self.run_counties = self.location_set_version_id in [841, 920]

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return {k: v for k, v in utilities.asdict(self).items() if k != 'run_counties'}


class PreprocessingSpecification(utilities.Specification):

    def __init__(self,
                 data: PreprocessingData,
                 workflow: PreprocessingWorkflowSpecification):
        self._data = data
        self._workflow = workflow

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': PreprocessingData,
            'workflow': PreprocessingWorkflowSpecification,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            spec_dict = utilities.filter_to_spec_fields(
                spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**spec_dict)
        return tuple(sub_specs.values())

    @property
    def data(self) -> PreprocessingData:
        return self._data

    @property
    def workflow(self) -> PreprocessingWorkflowSpecification:
        return self._workflow

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
        }
        return spec
