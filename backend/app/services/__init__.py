# Copyright (c) 2025 PronoTurf AI. All rights reserved.
# This source code is proprietary and confidential.
# Unauthorized copying, modification, distribution, or derivative works are strictly prohibited without prior written consent.

"""Services package with lazy imports."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Dict

__all__ = ["PMUService", "WeatherService", "pmu_service", "weather_service"]

_loaded_modules: Dict[str, ModuleType] = {}
_symbol_to_module = {
    "PMUService": "pmu_service",
    "WeatherService": "weather_service",
}


def _load_module(name: str) -> ModuleType:
    if name not in _loaded_modules:
        _loaded_modules[name] = import_module(f"{__name__}.{name}")
    return _loaded_modules[name]


def __getattr__(name: str):  # pragma: no cover
    if name in ("pmu_service", "weather_service"):
        return _load_module(name)
    module_name = _symbol_to_module.get(name)
    if module_name:
        module = _load_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from .pmu_service import PMUService
    from .weather_service import WeatherService