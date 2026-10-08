from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from .paths import CONFIG_ROOT


class ConfigNode(Mapping[str, Any]):
    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = {
            key: ConfigNode(value) if isinstance(value, Mapping) else value
            for key, value in values.items()
        }

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __getattr__(self, key: str) -> Any:
        try:
            return self._values[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in self._values.items():
            result[key] = value.to_dict() if isinstance(value, ConfigNode) else value
        return result


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是字典: {path}")
    return data


def _load_inherited_values(path: Path, loading: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in loading:
        chain = " -> ".join(str(item) for item in (*loading, path))
        raise ValueError(f"配置文件存在循环继承: {chain}")
    values = _read_yaml(path)
    base_name = values.pop("继承", None)
    if not base_name:
        return values
    base_path = path.parent / str(base_name)
    base_values = _load_inherited_values(base_path, (*loading, path))
    return _deep_merge(base_values, values)


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path)
    if not path.is_absolute():
        path = CONFIG_ROOT / path
    values = _load_inherited_values(path)
    values["_配置文件"] = str(path.resolve())
    return ConfigNode(values)
