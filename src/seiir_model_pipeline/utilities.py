from __future__ import annotations
import abc
from dataclasses import asdict as asdict_
from pathlib import Path
from typing import Dict, Union, Optional, Tuple
from pprint import pformat

from covid_shared import paths
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


def get_version(cli_argument: Optional[str], specification_value: Optional[str]) -> Path:
    """Determine the version to use hierarchically.

    CLI args override spec args.  Spec args override the default 'best'.

    """
    if cli_argument:
        version = Path(cli_argument).resolve()
    elif specification_value:
        version = specification_value
    else:
        version = paths.BEST_LINK
    return Path(version)


def get_output_root(cli_argument: Optional[str], specification_value: Optional[str],
                    default: Union[str, Path]) -> Path:
    """Determine the output root hierarchically.

    CLI arguments override specification args.  Specification args override
    the default.

    """
    if cli_argument:
        output_root = cli_argument
    elif specification_value:
        output_root = specification_value
    else:
        output_root = default
    return Path(output_root).resolve()
