import shlex
import os
from argparse import ArgumentParser, Namespace
from typing import Optional
import logging
import getpass

from jobmon.client.swarm.workflow.bash_task import BashTask
from jobmon.client.swarm.workflow.workflow import Workflow

from seiir_model_pipeline.core.utils import clone_run
from seiir_model_pipeline.core.versioner import BASE_DIR, Directories, OUTPUT_DIR, load_ode_settings, INPUT_DIR
from seiir_model_pipeline.executor.run import run
from seiir_model_pipeline.core.utils import load_locations
from seiir_model_pipeline.core.workflow import PROJECT
from seiir_model_pipeline.core.task import ExecParams

log = logging.getLogger(__name__)

VALIDATION_INPUT_DIR = BASE_DIR / 'seir-validations'
EXEC_PARAMS = ExecParams
EXEC_PARAMS.adjust({'m_mem_free': '5G'})


def parse_arguments(argstr: Optional[str] = None) -> Namespace:
    """
    Gets arguments from the command line or command line string.
    """
    log.info("parsing arguments")
    parser = ArgumentParser()
    parser.add_argument("--version-name", type=str, required=True)
    parser.add_argument("--time-holdout", type=int, required=True)
    parser.add_argument("--validation-output-dir", type=str, required=False)

    if argstr is not None:
        arglist = shlex.split(argstr)
        args = parser.parse_args(arglist)
    else:
        args = parser.parse_args()
    return args


def get_validation_version_name(original_version, time_holdout):
    new_version = f'{original_version}.validation.HO{time_holdout}'
    dirs = os.listdir(OUTPUT_DIR / 'ode')
    match = [x for x in dirs if f'validation.HO{time_holdout}' in x]
    version = len(match) + 1
    new_version += f'.{version:02}'
    return new_version


class InfectionSplitTask(BashTask):
    def __init__(self, old_directory, new_directory, location_id, time_holdout):

        command = (
            "split_infectionator " +
            f"--input-dir {old_directory} " +
            f"--output-dir {new_directory} " +
            f"--location-id {location_id} " +
            f"--time-holdout {time_holdout}"
        )

        super().__init__(
            command=command,
            name=f'seiir_validate_split_{location_id}',
            executor_parameters=EXEC_PARAMS,
            max_attempts=1,
        )


def create_infection_split_workflow(directories, old_directory, new_directory, time_holdout):
    """
    Create a workflow that splits the infectionator into train-test sets based
    on the time_holdout parameter.

    :param directories:
    :param old_directory:
    :param new_directory:
    :param time_holdout: (int)
    :return:
    """
    user = getpass.getuser()

    locations = load_locations(directories)
    tasks = [InfectionSplitTask(
        old_directory=old_directory,
        new_directory=new_directory,
        location_id=loc,
        time_holdout=time_holdout
    ) for loc in locations]

    wf = Workflow(
        workflow_args=f'seiir-train-test-split {old_directory} {new_directory}',
        project=PROJECT,
        stderr=f'/ihme/temp/sgeoutput/{user}/errors',
        stdout=f'/ihme/temp/sgeoutput/{user}/output',
        working_dir=f'/ihme/homes/{user}',
        seconds_until_timeout=60*60*24,
        resume=True
    )
    wf.add_tasks(tasks)
    return wf


def process_input_files(version_name, time_holdout):

    directories = Directories(ode_version=version_name)
    infection_version = load_ode_settings(version_name).infection_version
    new_infection_version = f'{infection_version}.validate.{time_holdout}'

    old_infection_directory = INPUT_DIR / infection_version
    new_infection_directory = VALIDATION_INPUT_DIR / new_infection_version
    if not os.path.exists(new_infection_directory):

        split_workflow = create_infection_split_workflow(
            directories=directories,
            old_directory=old_infection_directory,
            new_directory=new_infection_directory,
            time_holdout=time_holdout
        )
        exit_status = split_workflow.run()
        if exit_status != 0:
            raise RuntimeError("Error in the split workflow.")

    return new_infection_version


def launch_validation(version_name, time_holdout):
    log.info(f"Cloning {version_name} for a validation run with {time_holdout} holdout days.")

    new_infection_version = process_input_files(version_name, time_holdout)
    new_version_name = get_validation_version_name(
        original_version=version_name, time_holdout=time_holdout
    )
    clone_run(
        old_version=version_name,
        new_version=new_version_name,
        infection_dir=VALIDATION_INPUT_DIR,
        infection_version=new_infection_version
    )
    run(
        ode_version=new_version_name,
        regression_version=new_version_name,
        forecast_version=new_version_name,
        run_splicer=True,
        create_diagnostics=True
    )


def run_validation_analysis(version_name, output_path):
    pass


def main():

    args = parse_arguments()
    launch_validation(
        version_name=args.version_name,
        time_holdout=args.time_holdout
    )
    if isinstance(args.validation_output_dir, str):
        run_validation_analysis(
            version_name=args.version_name,
            output_path=args.validation_output_dir
        )


if __name__ == '__main__':
    main()
