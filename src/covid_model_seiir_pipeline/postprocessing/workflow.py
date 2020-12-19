import shutil
from typing import List

from covid_model_seiir_pipeline.workflow_tools.template import (
    TaskTemplate,
    WorkflowTemplate,
)
from covid_model_seiir_pipeline.postprocessing.specification import (
    POSTPROCESSING_JOBS
)


class PostprocessingTaskTemplate(TaskTemplate):

    task_name_template = "{measure}_{scenario}_post_processing"
    command_template = (
            f"{shutil.which('postprocess')} " +
            "--forecast-version {forecast_version} "
            "--scenario-name {scenario} "
            "--measure {measure}"
    )


class PostprocessingWorkflow(WorkflowTemplate):
    workflow_name_template = 'seiir-postprocess-{version}'
    task_template_classes = {
        POSTPROCESSING_JOBS.postprocess: PostprocessingTaskTemplate,
    }

    def attach_tasks(self, measures: List[str], scenario: str) -> None:
        postprocessing_template = self.task_templates[POSTPROCESSING_JOBS.postprocess]

        for measure in measures:
            postprocessing_task = postprocessing_template.get_task(
                forecast_version=self.version,
                scenario=scenario,
                measure=measure,
            )
            self.workflow.add_task(postprocessing_task)
