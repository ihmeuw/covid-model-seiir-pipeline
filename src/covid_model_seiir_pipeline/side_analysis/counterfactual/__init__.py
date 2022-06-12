from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import (
    COUNTERFACTUAL_JOBS,
    CounterfactualSpecification,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.data import (
    CounterfactualDataInterface
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.task import (
    counterfactual_scenario,
)
from covid_model_seiir_pipeline.side_analysis.counterfactual.main import (
    do_counterfactual,
    counterfactual,
)

SPECIFICATION = CounterfactualSpecification
COMMAND = counterfactual
APPLICATION_MAIN = do_counterfactual
TASKS = {
    COUNTERFACTUAL_JOBS.counterfactual_scenario: counterfactual_scenario,
}
