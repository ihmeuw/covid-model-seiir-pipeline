import pkgutil

import click

from covid_model_seiir_pipeline import pipeline
from covid_model_seiir_pipeline import side_analysis


@click.group()
def stask():
    """Parent command for individual seiir pipeline tasks."""
    pass


# Loops over every pipeline stage and adds all tasks to the `stask` cli group.
for package in [pipeline, side_analysis]:
    for importer, modname, is_pkg in pkgutil.iter_modules(package.__path__):
        if is_pkg:
            for task_name, task in __import__(modname).TASKS.items():
                stask.add_command(task, name=task_name)
