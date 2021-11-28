import shutil
from typing import List

from covid_shared import workflow

import covid_model_seiir_pipeline
from covid_model_seiir_pipeline.pipeline.preprocessing.specification import PREPROCESSING_JOBS


class PreprocessMeasureTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{PREPROCESSING_JOBS.preprocess_measure}_{{measure}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{PREPROCESSING_JOBS.preprocess_measure} "
            "--preprocessing-version {preprocessing_version} "
            "--measure {measure} "
            "-vv"
    )
    node_args = ['measure']
    task_args = ['preprocessing_version']


class PreprocessVaccineTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{PREPROCESSING_JOBS.preprocess_vaccine}_{{scenario}}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{PREPROCESSING_JOBS.preprocess_vaccine} "
            "--preprocessing-version {preprocessing_version} "
            "--scenario {scenario} "
            "-vv"
    )
    node_args = ['scenario']
    task_args = ['preprocessing_version']


class PreprocessSerologyTaskTemplate(workflow.TaskTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    task_name_template = f"{PREPROCESSING_JOBS.preprocess_vaccine}"
    command_template = (
            f"{shutil.which('stask')} "
            f"{PREPROCESSING_JOBS.preprocess_serology} "
            "--preprocessing-version {preprocessing_version} "            
            "-vv"
    )
    node_args = []
    task_args = ['preprocessing_version']


class PreprocessingWorkflow(workflow.WorkflowTemplate):
    tool = workflow.get_jobmon_tool(covid_model_seiir_pipeline)
    workflow_name_template = 'seiir-preprocessing-{version}'
    task_template_classes = {
        PREPROCESSING_JOBS.preprocess_measure: PreprocessMeasureTaskTemplate,
        PREPROCESSING_JOBS.preprocess_vaccine: PreprocessVaccineTaskTemplate,
        PREPROCESSING_JOBS.preprocess_serology: PreprocessSerologyTaskTemplate,
    }
    # Jobs here are not homogeneous so it's useful to get all failures if
    # things do fail.
    fail_fast = False

    def attach_tasks(self, measures: List[str], scenarios: List[str]) -> None:
        measure_template = self.task_templates[PREPROCESSING_JOBS.preprocess_measure]
        vaccine_template = self.task_templates[PREPROCESSING_JOBS.preprocess_vaccine]
        serology_template = self.task_templates[PREPROCESSING_JOBS.preprocess_serology]

        for measure in measures:
            task = measure_template.get_task(
                preprocessing_version=self.version,
                measure=measure,
            )
            self.workflow.add_task(task)
        for scenario in scenarios:
            task = vaccine_template.get_task(
                preprocessing_version=self.version,
                scenario=scenario,
            )
            self.workflow.add_task(task)
        task = serology_template.get_task(preprocessing_version=self.version)
        self.workflow.add_task(task)
