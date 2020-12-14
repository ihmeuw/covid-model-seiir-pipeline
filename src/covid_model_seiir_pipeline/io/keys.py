"""Primitives for representing logical datasets and groups for I/O."""
from pathlib import Path
from typing import NamedTuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .data_roots import DataRoot


class __LeafTemplates(NamedTuple):
    """Templates for leaf nodes in the on disk representation of data.

    The structure of a leaf node depends on the on disk format of the data.
    For csv files, a leaf node might be a file. For hdfs, on the other hand,
    it is an actual node enclosed in a group hierarchy in the hdf file.

    """
    DRAW_TEMPLATE: str = 'draw_{draw_id}'
    MEASURE_TEMPLATE: str = '{measure}'
    COV_SCENARIO_TEMPLATE: str = '{covariate_scenario}_scenario'
    COV_INFO_TEMPLATE: str = '{info_type}_info'


LEAF_TEMPLATES = __LeafTemplates()


class __PrefixTemplates(NamedTuple):
    """Templates for grouping prefixes in the on disk representation of data.

    Prefixes can be used to organize multiple versions of datasets under a
    single data root. A path-like representation of data would then look like
    `/root/prefix/...`.

    """
    SCENARIO_TEMPLATE: str = '{scenario}'


PREFIX_TEMPLATES = __PrefixTemplates()


class DatasetKey(NamedTuple):
    # noinspection PyUnresolvedReferences
    """Struct representing metadata for an on-disk dataset.

    A path-like representation of the data would be something like
    `/root/prefix/data_type/leaf_name`.

    Parameters
    ==========
    root
        The path to the parent directory in which this dataset lives or will
        live.
    disk_format
        The format of the dataset on disk.
    data_type
        The type of data being stored. This should be a logical type like
        `deaths` or `ode_parameters` rather than a structural type like
        `float`.
    leaf_name
        The name of the outer node representing a dataset. E.g. if storing
        data sets by draw, this might be `draw_27`.
    prefix
        An optional grouping prefix for high level partitioning of the data
        if required.

    """
    root: Path
    disk_format: str
    data_type: str
    leaf_name: str
    prefix: Optional[str]


class MetadataKey(NamedTuple):
    # noinspection PyUnresolvedReferences
    """Struct representing metadata for an on-disk metadata.

    Mo meta mo problems.

    Metadata is assumed to live at the top level of a process output so
    a path-like representation would look like `/root/data_type`

    Parameters
    ==========
    root
        The path to the parent directory in which this dataset lives or will
        live.
    disk_format
        The format of the dataset on disk.
    data_type
        The type of metadata being stored. This should be a logical type like
        `model_specification` or `locations` rather than a structural type
        like `float`.

    """
    root: Path
    disk_format: str
    data_type: str


class DatasetType:
    """Factory for DatasetKeys."""

    def __init__(self,
                 name: str,
                 leaf_template: str,
                 prefix_template: str = None,
                 root: Path = None,
                 disk_format: str = None):
        self.name = name
        if prefix_template is not None and prefix_template not in PREFIX_TEMPLATES:
            raise ValueError(f'Invalid prefix_template specification: {prefix_template} for DatasetType {self.name}. '
                             f'prefix_template must be one of {PREFIX_TEMPLATES}.')
        self.prefix_template = prefix_template
        if leaf_template not in LEAF_TEMPLATES:
            raise ValueError(f'Invalid leaf_template specification: {leaf_template} for DatasetType {self.name}. '
                             f'leaf_template must be one of {LEAF_TEMPLATES}.')
        self.leaf_template = leaf_template

        self.root = root
        # Disk format validated upstream
        self.disk_format = disk_format

    def __get__(self, instance: 'DataRoot', owner=None) -> 'DatasetType':
        return type(self)(self.name, self.leaf_template, self.prefix_template, instance._root, instance._data_format)

    def __call__(self, *args, **key_kwargs) -> DatasetKey:
        assert self.root is not None, f'DatasetType must be bound to a DataRoot object to be called.'
        assert self.disk_format is not None, 'DatasetType must be bound to a DataRoot object to be called.'

        if args:
            raise TypeError('All dataset key arguments must be specified as keyword arguments. '
                            f'You specified args {args} and kwargs {key_kwargs} to {self.name}.')

        leaf_name = self.leaf_template.format(**key_kwargs)
        path_name = self.prefix_template.format(**key_kwargs) if self.prefix_template else None
        return DatasetKey(self.root, self.disk_format, self.name, leaf_name, path_name)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(["=".join([k, str(v)]) for k, v in self.__dict__.items()])})'


class MetadataType:
    def __init__(self, name: str,
                 root: Path = None,
                 disk_format: str = None):
        self.name = name
        self.root = root
        # Disk format validated upstream
        self.disk_format = disk_format

    def __get__(self, instance: 'DataRoot', owner=None) -> 'MetadataType':
        return type(self)(self.name, instance._root, instance._metadata_format)

    def __call__(self, *args, **key_kwargs) -> MetadataKey:
        assert self.root is not None, f'MetadataType must be bound to a DataRoot object to be called.'
        assert self.disk_format is not None, 'MetadataType must be bound to a DataRoot object to be called.'

        if args or key_kwargs:
            raise TypeError('Metadata keys have no parameterization. You provided '
                            f'args {args} and key_kwargs {key_kwargs} for {self.name}.')

        return MetadataKey(self.root, self.disk_format, self.name)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f'{type(self).__name__}({", ".join(["=".join([k, str(v)]) for k, v in self.__dict__.items()])})'
