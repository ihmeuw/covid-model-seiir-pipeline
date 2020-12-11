"""Primitives for representing logical datasets and groups for I/O.


"""

from pathlib import Path
from typing import NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_roots import DataRoot


class __LeafTemplates(NamedTuple):
    """Templates for leaf nodes in the on disk representation of datasets.

    The structure of a leaf node depends on the on disk format of the data.
    For csvs, a leaf node might be a file.  For hdfs, on the other hand,
    it is an actual node enclosed in a hierarchy in the hdf file.

    """
    DRAW_TEMPLATE: str = 'draw_{draw_id}'
    MEASURE_TEMPLATE: str = '{measure}'
    COV_SCENARIO_TEMPLATE: str = '{covariate_scenario}_scenario'
    COV_INFO_TEMPLATE: str = '{info_type}_info'


LEAF_TEMPLATES = __LeafTemplates()


class __PathTemplates(NamedTuple):
    SCENARIO_TEMPLATE: str = '{scenario}'


PATH_TEMPLATES = __PathTemplates()


class DatasetKey(NamedTuple):
    """Struct representing metadata for an on-disk dataset.

    Attributes
    ==========
    root
        The path to the parent directory in which this dataset lives or will
        live.
    output_format
        The format of the dataset on disk.
    data_type
        The type of data being stored. This should be a logical type like
        `deaths` or `ode_parameters` rather than a structural type like
        `float`.
    leaf_template
        A string template to be filled by parameters that represents a single
        dataset.
    key_args
        Key-value pairs to populate the leaf_template and determine nesting
        of a dataset.

    """
    root: Path
    disk_format: str
    data_type: str
    leaf_name: str
    path_name: Optional[str]


class MetadataKey(NamedTuple):
    root: Path
    disk_format: str
    data_type: str


class DatasetType:

    def __init__(self, name: str,
                 leaf_template: str, path_template: str = None,
                 root: Path = None, disk_format: str = None):
        self.name = name
        if path_template is not None and path_template not in PATH_TEMPLATES:
            raise ValueError(f'Invalid path_template specification: {path_template} for DatasetType {self.name}. '
                             f'path_template must be one of {PATH_TEMPLATES}.')
        self.path_template = path_template
        if leaf_template not in LEAF_TEMPLATES:
            raise ValueError(f'Invalid leaf_template specification: {leaf_template} for DatasetType {self.name}. '
                             f'leaf_template must be one of {LEAF_TEMPLATES}.')
        self.leaf_template = leaf_template

        self.root = root
        # Disk format validated upstream
        self.disk_format = disk_format

    def __get__(self, instance: 'DataRoot', owner=None) -> 'DatasetType':
        return type(self)(self.name, self.leaf_template, self.path_template, instance._root, instance._data_format)

    def __call__(self, *args, **key_kwargs) -> DatasetKey:
        if args:
            raise ValueError

        leaf_name = self.leaf_template.format(**key_kwargs)
        path_name = self.path_template.format(**key_kwargs) if self.path_template else None
        return DatasetKey(self.root, self.disk_format, self.name, leaf_name, path_name)

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(["=".join([k, str(v)]) for k, v in self.__dict__.items()])})'


class MetadataType:
    def __init__(self, name: str,
                 root: Path = None, disk_format: str = None):
        self.name = name
        self.root = root
        # Disk format validated upstream
        self.disk_format = disk_format

    def __get__(self, instance: 'DataRoot', owner=None) -> 'MetadataType':
        return type(self)(self.name, instance._root, instance._metadata_format)

    def __call__(self, *args, **key_kwargs) -> MetadataKey:
        if args:
            raise

        return MetadataKey(self.root, self.disk_format, self.name)

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(["=".join([k, str(v)]) for k, v in self.__dict__.items()])})'
