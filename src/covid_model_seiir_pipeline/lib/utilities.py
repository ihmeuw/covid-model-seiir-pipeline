from __future__ import annotations

import abc
from bdb import BdbQuit
from dataclasses import asdict as asdict_
import functools
from pathlib import Path
from typing import Any, Callable, Dict, Union, Optional, Tuple
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


def get_argument_hierarchically(cli_argument: Optional,
                                specification_value: Optional,
                                default: Any) -> Any:
    """Determine the argument to use hierarchically.

    Prefer cli args over values in a specification file over the default.
    """
    if cli_argument:
        output = cli_argument
    elif specification_value:
        output = specification_value
    else:
        output = default
    return output


def get_input_root(cli_argument: Optional[str], specification_value: Optional[str],
                   last_stage_root: Union[str, Path]) -> Path:
    """Determine the version to use hierarchically.

    CLI args override spec args.  Spec args override the default 'best'.

    """
    version = get_argument_hierarchically(cli_argument, specification_value, paths.BEST_LINK)
    root = cli_tools.get_last_stage_directory(version, last_stage_root=last_stage_root)
    return root.resolve()


def get_location_info(location_specification: Optional[str],
                      spec_lsvid: Optional[int],
                      spec_location_file: Optional[str]) -> Tuple[int, str]:
    """Resolves a location specification from the cli args and run spec.

    Parameters
    ----------
    location_specification
        Either a location set version  id or a path to a location
        hierarchy file.
    spec_lsvid
        The location set version id provided in the run spec.
    spec_location_file
        The location file provided in the run spec.

    Returns
    -------
        A valid lsvid and location file specification constructed from the
        input arguments.  CLI args take precedence over run spec parameters.
        If nothing is provided, return 0 and '' for the lsvid and location
        file, respectively, and let the model stage code set sensible
        default operations.

    """
    if spec_lsvid and spec_location_file:
        raise ValueError('Both a location set version id and a location file were provided in '
                         'the specification. Only one option may be used.')
    location_specification = get_argument_hierarchically(location_specification, spec_lsvid, 0)
    location_specification = get_argument_hierarchically(location_specification, spec_location_file, '')
    try:
        lsvid = int(location_specification)
        location_file = ''
    except ValueError:  # The command line argument was a path
        lsvid = 0
        location_file = location_specification
    return lsvid, location_file


def get_output_root(cli_argument: Optional[str], specification_value: Optional[str]) -> Path:
    """Determine the output root hierarchically.

    CLI arguments override specification args.  Specification args override
    the default.

    """
    # Default behavior handled by CLI.
    version = get_argument_hierarchically(cli_argument, specification_value, cli_argument)
    version = Path(version).resolve()
    return version


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


def handle_exceptions(func: Callable, logger: Any, with_debugger: bool) -> Callable:
    """Drops a user into an interactive debugger if func raises an error."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (BdbQuit, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.exception("Uncaught exception {}".format(e))
            if with_debugger:
                import pdb
                import traceback
                traceback.print_exc()
                pdb.post_mortem()
            else:
                raise

    return wrapped
