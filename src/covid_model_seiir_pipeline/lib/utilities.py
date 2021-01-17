from __future__ import annotations

import abc
from dataclasses import asdict as asdict_
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple
from pprint import pformat

from covid_shared import paths, shell_tools, cli_tools
import numpy as np
import pandas as pd
import yaml

from covid_model_seiir_pipeline.lib.ihme_deps import  get_location_metadata


class YamlIOMixin:
    """Mixin for reading and writing yaml files."""

    @staticmethod
    def _coerce_path(path: Union[str, Path]) -> Path:
        path = Path(path)
        if path.suffix not in ['.yaml', '.yml']:
            raise ValueError('Path must point to a yaml file. '
                             f'You provided {str(path)}')
        return path

    @classmethod
    def _load(cls, path: Union[str, Path]):
        path = cls._coerce_path(path)
        with path.open() as f:
            data = yaml.full_load(f)

        return data

    @classmethod
    def _dump(cls, data, path: Union[str, Path]) -> None:
        path = cls._coerce_path(path)
        with path.open('w') as f:
            yaml.dump(data, f, sort_keys=False)


class Specification(YamlIOMixin):
    """Generic class for pipeline stage specifications."""

    @classmethod
    def from_path(cls, specification_path: Union[str, Path]) -> Specification:
        """Builds the specification from a file path."""
        spec_dict = cls._load(specification_path)
        return cls.from_dict(spec_dict)

    @classmethod
    def from_dict(cls, spec_dict: Dict) -> Specification:
        """Builds the specification from a dictionary."""
        args = cls.parse_spec_dict(spec_dict)
        return cls(*args)

    @classmethod
    @abc.abstractmethod
    def parse_spec_dict(cls, specification: Dict) -> Tuple:
        """Parses a dict representation of the specification into init args."""
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> Dict:
        """Coerce the specification to a dict."""
        raise NotImplementedError

    def dump(self, path: Union[str, Path]) -> None:
        """Writes this specification to a file."""
        data = self.to_dict()
        self._dump(data, path)

    def __repr__(self):
        return f'{self.__class__.__name__}(\n{pformat(self.to_dict())}\n)'


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













def make_log_dirs(output_dir: Union[str, Path], prefix: str = None) -> Tuple[str, str]:
    """Create log directories in output root and return the paths."""
    log_dir = Path(output_dir) / 'logs'
    if prefix:
        log_dir /= prefix
    std_out = log_dir / 'output'
    std_err = log_dir / 'error'
    shell_tools.mkdir(std_out, exists_ok=True, parents=True)
    shell_tools.mkdir(std_err, exists_ok=True, parents=True)

    return str(std_out), str(std_err)


def load_location_hierarchy(location_set_id: int = None,
                            location_set_version_id: int = None,
                            location_file: Path = None):
    ids = location_set_id and location_set_version_id
    assert (ids and not location_file) or (not ids and location_file)

    if ids:
        # Hide this import so the code stays portable outside IHME by using
        # a locations file directly.
        try:
            return get_location_metadata(location_set_id=location_set_id,
                                         location_set_version_id=location_set_version_id)
        except ValueError:
            return get_location_metadata(location_set_id=location_set_id,
                                         location_set_version_id=location_set_version_id,
                                         gbd_round_id=6)
    else:
        return pd.read_csv(location_file)
