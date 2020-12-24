import itertools
import shutil
from typing import List

from covid_model_seiir_pipeline.lib import workflow
from covid_model_seiir_pipeline.pipeline.postprocessing.specification import (
    POSTPROCESSING_JOBS,
)


class PostprocessingTaskTemplate(workflow.TaskTemplate):

    task_name_template = f"{POSTPROCESSING_JOBS.postprocess}_{{scenario}}_{{measure}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{POSTPROCESSING_JOBS.postprocess} " +
            "--postprocessing-version {postprocessing_version} "
            "--scenario {scenario} "
            "--measure {measure}"
    )
    node_args = ['scenario', 'measure']
    task_args = ['postprocessing_version']


class ResampleMapTaskTemplate(workflow.TaskTemplate):
    task_name_template = f"{POSTPROCESSING_JOBS.resample}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{POSTPROCESSING_JOBS.resample} " +
            "--postprocessing-version {postprocessing_version} "
    )
    node_args = []
    task_args = ['postprocessing_version']


class PostprocessingWorkflow(workflow.WorkflowTemplate):
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
