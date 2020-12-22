import itertools
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
            "--postprocessing-version {postprocessing_version} "
            "--scenario-name {scenario} "
            "--measure {measure}"
    )


class ResampleMapTaskTemplate(TaskTemplate):
    task_name_template = "seiir_resample_map"
    command_template = (
            f"{shutil.which('resample_map')} " +
            "--postprocessing-version {postprocessing_version} "
    )


class PostprocessingWorkflow(WorkflowTemplate):
    workflow_name_template = 'seiir-postprocess-{version}'
    task_template_classes = {
        POSTPROCESSING_JOBS.resample: ResampleMapTaskTemplate,
        POSTPROCESSING_JOBS.postprocess: PostprocessingTaskTemplate,
    }

    def attach_tasks(self, measures: List[str], scenarios: List[str]) -> None:
        resample_template = self.task_templates[POSTPROCESSING_JOBS.resample]
        postprocessing_template = self.task_templates[POSTPROCESSING_JOBS.postprocess]

        # The draw resampling map is produced for one reference scenario
        # after the forecasts and then used to postprocess all measures for
        # all scenarios.
        resample_task = resample_template.get_task(
            postprocessing_version=self.version
        )
        self.workflow.add_task(resample_task)

        for measure, scenario in itertools.product(measures, scenarios):
            postprocessing_task = postprocessing_template.get_task(
                postprocessing_version=self.version,
                scenario=scenario,
                measure=measure,
            )
            postprocessing_task.add_upstream(resample_task)
            self.workflow.add_task(postprocessing_task)
