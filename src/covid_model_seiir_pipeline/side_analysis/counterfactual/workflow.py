import itertools
import shutil
from typing import Iterable

from covid_shared import workflow

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.side_analysis.counterfactual.specification import COUNTERFACTUAL_JOBS


class CounterfactualTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f'{COUNTERFACTUAL_JOBS.counterfactual_scenario}_{{scenario}}'
    command_template = (
        f"{shutil.which('stask')} "
        f"{COUNTERFACTUAL_JOBS.counterfactual} "
        "--counterfactual-version {counterfactual_version} "
        "--scenario {scenario} "
        "--draw-id {draw_id} "
        "-vv"
    )
    node_args = ['scenario', 'draw_id']
    task_args = ['counterfactual_version']


class CounterfactualWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-counterfactual-{version}'
    task_template_classes = {
        COUNTERFACTUAL_JOBS.counterfactual_scenario: CounterfactualTaskTemplate,
    }

    def attach_tasks(self, n_draws: int, scenarios: Iterable[str]):
        counterfactual_template = self.task_templates[COUNTERFACTUAL_JOBS.counterfactual]

        for scenario_name, draw in itertools.product(scenarios, range(n_draws)):
            counterfactual_task = counterfactual_template.get_task(
                counterfactual_version=self.version,
                scenario=scenario_name,
                draw_id=draw,
            )
            self.workflow.add_task(counterfactual_task)
