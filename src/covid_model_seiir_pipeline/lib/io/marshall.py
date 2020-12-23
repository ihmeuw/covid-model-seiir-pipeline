from contextlib import contextmanager
import io
import os
from pathlib import Path
from typing import Any, Tuple
import zipfile

from covid_shared.shell_tools import mkdir
import pandas as pd
import yaml

from covid_model_seiir_pipeline.lib.io.keys import (
    DatasetKey,
    MetadataKey,
)


class CSVMarshall:
    """
    Marshalls DataFrames to/from CSV files.

    This implementation directly mirrors existing behavior but does so within a
    new marshalling interface.
    """
    # interface methods
    @classmethod
    def dump(cls, data: pd.DataFrame, key: DatasetKey, strict: bool = True) -> None:
        path = cls._resolve_key(key)

        if strict and path.exists():
            msg = f"Cannot dump data for key {key} - would overwrite"
            raise LookupError(msg)

        data.to_csv(path, index=False)

    @classmethod
    def load(cls, key: DatasetKey) -> pd.DataFrame:
        path = cls._resolve_key(key)
        return pd.read_csv(path)

    @classmethod
    def touch(cls, *paths: Path) -> None:
        for path in paths:
            mkdir(path, parents=True, exists_ok=True)

    @classmethod
    def exists(cls, key: DatasetKey) -> bool:
        path = cls._resolve_key(key)
        return path.exists()

    @classmethod
    def _resolve_key(cls, key: DatasetKey) -> Path:
        path = key.root
        if key.prefix:
            path /= key.prefix
        path = path / key.data_type / key.leaf_name
        return path.with_suffix(".csv")


class ZipMarshall:
    # interface methods

    @classmethod
    def dump(cls, data: pd.DataFrame, key: DatasetKey, strict: bool = True) -> None:
        zip_path, node = cls._resolve_key(key)
        with zipfile.ZipFile(zip_path, mode='a') as container:
            with cls._open_node(container, node, strict) as outf:
                # stream writes through a wrapper that does str => bytes conversion
                # chunksize denotes number of rows to write at once. it is not
                # clear to me that 5 is at all a good number
                wrapper = io.TextIOWrapper(outf, write_through=True)
                data.to_csv(wrapper, index=False, chunksize=5)

    @classmethod
    def load(cls, key: DatasetKey) -> pd.DataFrame:
        zip_path, node = cls._resolve_key(key)
        with zipfile.ZipFile(zip_path) as container:
            with container.open(node) as inf:
                return pd.read_csv(inf)

    @classmethod
    def exists(cls, key: DatasetKey) -> bool:
        zip_path, node = cls._resolve_key(key)
        with zipfile.ZipFile(zip_path) as container:
            try:
                container.getinfo(node)
                return True
            except KeyError:
                return False

    @classmethod
    def touch(cls, *paths: Path) -> None:
        for path in paths:
            mkdir(path.parent, parents=True, exists_ok=True)
            mode = 0o664
            old_umask = os.umask(0o777 - mode)
            try:
                path.with_suffix('.zip').touch()
            finally:
                os.umask(old_umask)

    @classmethod
    def _resolve_key(cls, key: DatasetKey) -> Tuple[Path, str]:
        zip_path = key.root
        if key.prefix:
            zip_path /= key.prefix
        zip_path /= key.data_type
        node = f'{key.leaf_name}.csv'
        return zip_path, node

    @classmethod
    @contextmanager
    def _open_node(cls, zip_container, node: str, strict: bool):
        try:
            zip_container.getinfo(node)
        except KeyError:
            pass  # file does not exist - everything OK
        else:
            if strict:
                raise LookupError(f"Cannot dump data for key {node} - would overwrite")

        with zip_container.open(node, 'w') as outf:
            yield outf


class HDF5Marshall:
    """
    Marshalls data to/from HDF5 format.

    HDF is a typed dataset format that brings several additional steps to
    successful marshalling.

    For the uninitiated, HDF5

    * stores all data in a single file
    * supports multiple sets of data (Dataset)
    * supports grouping of data (groups) which are analogous to directories

    """

    @classmethod
    def dump(cls, data, key, strict=True):
        hdf_path, node = cls._resolve_key(key)
        with pd.HDFStore(hdf_path) as container:
            if node in container and strict:
                raise LookupError(f"Cannot dump data for key {node} - would overwrite")
            container.put(node, data)

    @classmethod
    def load(cls, key):
        hdf_path, node = cls._resolve_key(key)
        with pd.HDFStore(hdf_path, mode='r') as container:
            try:
                return container.get(node)
            except KeyError:
                raise RuntimeError(f"No data set for {key} saved!")


    @classmethod
    def touch(cls, *paths: Path) -> None:
        for path in paths:
            mkdir(path.parent, parents=True, exists_ok=True)
            mode = 0o664
            old_umask = os.umask(0o777 - mode)
            try:
                path.with_suffix('.hdf').touch()
            finally:
                os.umask(old_umask)

    @classmethod
    def exists(cls, key):
        hdf_path, node = cls._resolve_key(key)
        with pd.HDFStore(hdf_path, mode='r') as container:
            return node in container

    @classmethod
    def _resolve_key(cls, key: DatasetKey):
        hdf_path = key.root
        if key.prefix:
            hdf_path /= key.prefix
        hdf_path /= key.data_type
        node = f'{key.leaf_name}'
        return str(hdf_path), node


class YamlMarshall:
    """Marshalls primitive python data structures to and from yaml."""

    @classmethod
    def dump(cls, data: Any, key: MetadataKey, strict: bool = True) -> None:
        path = cls._resolve_key(key)
        if strict and path.exists():
            msg = f"Cannot dump data for key {key} - would overwrite"
            raise LookupError(msg)

        with path.open('w') as file:
            yaml.dump(data, file)

    @classmethod
    def load(cls, key: MetadataKey) -> Any:
        path = cls._resolve_key(key)
        with path.open() as file:
            data = yaml.full_load(file)
        return data

    @classmethod
    def exists(cls, key: MetadataKey) -> bool:
        path = cls._resolve_key(key)
        return path.exists()

    @classmethod
    def _resolve_key(cls, key: MetadataKey) -> Path:
        path = (key.root / key.data_type).with_suffix(".yaml")
        return path


DATA_STRATEGIES = {
    'csv': CSVMarshall,
    'zip': ZipMarshall,
    'hdf': HDF5Marshall,
}
METADATA_STRATEGIES = {
    'yaml': YamlMarshall,
}
STRATEGIES = {**DATA_STRATEGIES, **METADATA_STRATEGIES}
