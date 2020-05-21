import abc
from dataclasses import asdict as asdict_
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
from pprint import pformat

import numpy as np
import yaml


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
    def from_path(cls, specification_path: Union[str, Path]) -> 'Specification':
        """Builds the specification from a file path."""
        spec_dict = cls._load(specification_path)
        return cls.from_dict(spec_dict)

    @classmethod
    def from_dict(cls, spec_dict: Dict) -> 'Specification':
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
