from typing import Any, Union

from .data_roots import DataRoot
from .keys import (
    DatasetKey,
    MetadataKey,
)
from .marshall import STRATEGIES


def load(key: Union[MetadataKey, DatasetKey]) -> Any:
    if key.disk_format not in STRATEGIES:
        raise
    return STRATEGIES[key.disk_format].load(key)


def dump(dataset: Any, key: Union[MetadataKey, DatasetKey]):
    if key.disk_format not in STRATEGIES:
        raise
    STRATEGIES[key.disk_format].dump(dataset, key)


def exists(key: Union[MetadataKey, DatasetKey]) -> bool:
    if key.disk_format not in STRATEGIES:
        raise
    return STRATEGIES[key.disk_format].exists(key)


def touch(data_root: DataRoot):
    # TODO:
    pass
