from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __CounterfactualJobs(NamedTuple):
    counterfactual_scenario: str


COUNTERFACTUAL_JOBS = __CounterfactualJobs(*__CounterfactualJobs._fields)


class CounterfactualTaskSpecification(workflow.TaskSpecification):
    default_max_runtime_seconds = 15000
    default_m_mem_free = '50G'
    default_num_cores = 11


class CounterfactualWorkflowSpecification(workflow.WorkflowSpecification):
    tasks = {
        COUNTERFACTUAL_JOBS.counterfactual_scenario: CounterfactualTaskSpecification,
    }


@dataclass
class CounterfactualData:
    seir_counterfactual_input_version: str = field(default='best')
    seir_fit_version: str = field(default='best')
    seir_forecast_version: str = field(default='best')
    output_root: str = field(default='')
    output_format: str = field(default='csv')

    def to_dict(self) -> Dict:
        return utilities.asdict(self)


@dataclass
class CounterfactualScenarioParameters:
    name: str = field(default='')
    beta: str = field(default='')
    initial_condition: str = field(default='')
    vaccine_coverage: str = field(default='')
    variant_prevalence: str = field(default='')
    mask_use: str = field(default='')
    mobility: str = field(default='')
    start_date: str = field(default='2019-01-01')

    def to_dict(self) -> Dict:
        return {k: v for k, v in utilities.asdict(self).items() if k != 'name'}


class CounterfactualSpecification(utilities.Specification):

    def __init__(self,
                 data: CounterfactualData,
                 workflow: CounterfactualWorkflowSpecification,
                 scenarios: List[CounterfactualScenarioParameters]):
        self._data = data
        self._workflow = workflow
        self._scenarios = {s.name: s for s in scenarios}

    @classmethod
    def parse_spec_dict(cls, spec_dict: Dict) -> Tuple:
        sub_specs = {
            'data': CounterfactualData,
            'workflow': CounterfactualWorkflowSpecification,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            sub_spec_dict = utilities.filter_to_spec_fields(
                spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**sub_spec_dict)

        # scenarios
        scenario_dicts = spec_dict.get('scenarios', {})
        scenarios = []
        for name, scenario_spec in scenario_dicts.items():
            scenario_spec = utilities.filter_to_spec_fields(scenario_spec, CounterfactualScenarioParameters())
            scenarios.append(CounterfactualScenarioParameters(name, **scenario_spec))
        sub_specs['scenarios'] = scenarios

        return tuple(sub_specs.values())

    @property
    def data(self) -> CounterfactualData:
        return self._data

    @property
    def workflow(self) -> CounterfactualWorkflowSpecification:
        return self._workflow

    @property
    def scenarios(self) -> Dict[str, CounterfactualScenarioParameters]:
        return self._scenarios

    def to_dict(self) -> Dict:
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'scenarios': {k: v.to_dict() for k, v in self._scenarios.items()},
        }
        return spec
