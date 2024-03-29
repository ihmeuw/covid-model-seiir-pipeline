from __future__ import annotations

import abc
from contextlib import contextmanager
from dataclasses import asdict as asdict_, fields, is_dataclass
import hashlib
import os
from pathlib import Path
import sys
from typing import Dict, Union, Tuple
from pprint import pformat

from covid_shared import ihme_deps, workflow
import inflection
import numpy as np
import pandas as pd
import yaml


class YamlIOMixin:
    """Mixin for reading and writing yaml files."""

    @staticmethod
    def _coerce_path(path: Union[str, Path]) -> Path:
        path = Path(path)
        path = Path(str(path).replace('/ihme/covid-19-2', '/ihme/covid-19'))
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
    def spec_file_name(cls) -> str:
        return f'{inflection.underscore(cls.__name__)}.yaml'

    @classmethod
    def from_version_root(cls, version_root: Union[str, Path]) -> Specification:
        path = Path(version_root) / cls.spec_file_name()
        return cls.from_path(path)

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


def filter_to_spec_fields(spec_dict: dict, specification):
    if is_dataclass(specification):
        return {
            k: v for k, v in spec_dict.items()
            if k in [f.name for f in fields(specification)]
        }
    else:
        return spec_dict


def load_location_hierarchy(location_set_version_id: int = None,
                            location_file: Path = None, **kwargs):
    assert ((location_set_version_id and not location_file)
            or (not location_set_version_id and location_file))

    if location_set_version_id:
        return ihme_deps.get_location_hierarchy_by_version(
            location_set_version_id=location_set_version_id,
        )
    else:
        return pd.read_csv(location_file)


def get_random_seed(key: str):
    # 4294967295 == 2**32 - 1 which is the maximum allowable seed for a `numpy.random.RandomState`.
    seed = int(hashlib.sha1(key.encode('utf8')).hexdigest(), 16) % 4294967295
    return seed


def get_random_state(key: str):
    seed = get_random_seed(key)
    random_state = np.random.RandomState(seed=seed)
    return random_state


@contextmanager
def suppress_output(filter_list):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StreamFilter(filter_list, sys.stdout)
    sys.stderr = StreamFilter(filter_list, sys.stderr)
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class StreamFilter(object):
    def __init__(self, strings_to_filter, stream):
        self.stream = stream
        self.triggered = False
        self.strings_to_filter = strings_to_filter

    def __getattr__(self, attr_name):
        return getattr(self.stream, attr_name)

    def write(self, data):
        if data == '\n' and self.triggered:
            self.triggered = False
        else:
            if data not in self.strings_to_filter:
                self.stream.write(data)
                self.stream.flush()
            else:
                # caught bad pattern
                self.triggered = True

    def flush(self):
        self.stream.flush()
