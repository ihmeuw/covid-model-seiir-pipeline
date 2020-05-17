from dataclasses import asdict as asdict_
from pathlib import Path
from typing import Dict, Union

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
