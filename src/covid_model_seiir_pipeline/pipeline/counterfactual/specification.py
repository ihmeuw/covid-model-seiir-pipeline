from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Tuple

from covid_shared import workflow

from covid_model_seiir_pipeline.lib import (
    utilities,
)


class __CounterfactualJobs(NamedTuple):
    counterfactual: str


COUNTERFACTUAL_JOBS = __CounterfactualJobs(*__CounterfactualJobs._fields)


class CounterfactualTaskSpecification(workflow.TaskSpecification):
    """Specification of execution parameters for counterfactual tasks."""
    default_max_runtime_seconds = 3000
    default_m_mem_free = '5G'
    default_num_cores = 1


class CounterfactualWorkflowSpecification(workflow.WorkflowSpecification):
    """Specification of execution parameters for regression workflows."""
    tasks = {
        COUNTERFACTUAL_JOBS.counterfactual: CounterfactualTaskSpecification,
    }


@dataclass
class CounterfactualData:
    """Specifies the inputs and outputs for a counterfactual run."""
    counterfactual_input_version: str = field(default='best')
    forecast_version: str = field(default='best')

    output_root: str = field(default='')
    output_format: str = field(default='csv')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists."""
        return utilities.asdict(self)


@dataclass
class CounterfactualScenarioParameters:
    """Specification for a counterfactual scenario"""

    name: str = field(default='')
    beta: str = field(default='reference')
    vaccine_coverage: str = field(default='reference')
    variant_prevalence: str = field(default='reference')
    start_date: str = field(default='2019-01-01')

    def to_dict(self) -> Dict:
        """Converts to a dict, coercing list-like items to lists.

        Drops the name parameter as it's used as a key in the specification.

        """
        return {k: v for k, v in utilities.asdict(self).items() if k != 'name'}


class CounterfactualSpecification(utilities.Specification):
    """Specification for a counterfactual run."""

    def __init__(self,
                 data: CounterfactualData,
                 workflow: CounterfactualWorkflowSpecification,
                 scenarios: List[CounterfactualScenarioParameters]):
        self._data = data
        self._workflow = workflow
        self._scenarios = {s.name: s for s in scenarios}

    @classmethod
    def parse_spec_dict(cls, counterfactual_spec_dict: Dict) -> Tuple:
        """Constructs a counterfactual specification from a dictionary."""
        sub_specs = {
            'data': CounterfactualData,
            'workflow': CounterfactualWorkflowSpecification,
        }
        for key, spec_class in list(sub_specs.items()):  # We're dynamically altering. Copy with list
            spec_dict = utilities.filter_to_spec_fields(
                counterfactual_spec_dict.get(key, {}),
                spec_class(),
            )
            sub_specs[key] = spec_class(**spec_dict)

        # scenarios
        scenario_dicts = counterfactual_spec_dict.get('scenarios', {})
        scenarios = []
        for name, scenario_spec in scenario_dicts.items():
            scenario_spec = utilities.filter_to_spec_fields(scenario_spec, CounterfactualScenarioParameters())
            scenarios.append(CounterfactualScenarioParameters(name, **scenario_spec))
        sub_specs['scenarios'] = scenarios

        return tuple(sub_specs.values())

    @property
    def data(self) -> CounterfactualData:
        """The data specification for the counterfactual."""
        return self._data

    @property
    def workflow(self) -> CounterfactualWorkflowSpecification:
        """The workflow specification for the counterfactual."""
        return self._workflow

    @property
    def scenarios(self) -> Dict[str, CounterfactualScenarioParameters]:
        """The counterfactual scenario parameters."""
        return self._scenarios

    def to_dict(self) -> Dict:
        """Converts the specification to a dict."""
        spec = {
            'data': self.data.to_dict(),
            'workflow': self.workflow.to_dict(),
            'covariates': {k: v.to_dict() for k, v in self._scenarios.items()},
        }
        return spec
