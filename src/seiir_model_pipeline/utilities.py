from dataclasses import asdict as asdict_
from pathlib import Path
from typing import Dict, Union, Optional

import numpy as np
import yaml


def load_specification(specification_path: Union[str, Path]):
    specification_path = Path(specification_path)
    if specification_path.suffix not in ['.yaml', '.yml']:
        raise ValueError('Specification must be a yaml file.')

    with specification_path.open() as specification_file:
        specification = yaml.full_load(specification_file)

    return specification


def asdict(data_class) -> Dict:
    """Type coerce items for easy serialization"""
    data = asdict_(data_class)
    out = {}
    for k, v in data.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


class VersionDirectory:

    def __init__(self,
                 version_name: Optional[str] = None,
                 version_dir: Optional[Union[str, Path]] = None,
                 root_dir: Union[str, Path] = Path()):

        if version_name is None and version_dir is None:
            raise ValueError("must specify either regression_version or regression_dir.")
        elif version_name is not None and version_dir is not None:
            raise ValueError("cannot specify both regression_version and regression_dir. Try "
                             "regression_version and regression_root together or "
                             "regression_dir alone.")

        if version_dir is None:
            self.version_name = str(version_name)
            self.version_dir = Path(root_dir) / self.version_name
        else:
            self.version_dir = Path(version_dir)

        # reassign everything based on regression_dir
        self.root_dir = self.version_dir.parent
        self.version_name = self.version_dir.name
