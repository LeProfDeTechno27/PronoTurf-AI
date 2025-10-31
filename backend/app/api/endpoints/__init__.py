"""API endpoints routers."""
from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Dict

__all__ = [
    "auth",
    "health",
    "hippodromes",
    "reunions",
    "courses",
    "analytics",
]

# Cache imported modules to avoid re-importing
_loaded_modules: Dict[str, ModuleType] = {}


def __getattr__(name: str) -> ModuleType:  # pragma: no cover - thin proxy
    if name in __all__:
        if name not in _loaded_modules:
            _loaded_modules[name] = import_module(f"{__name__}.{name}")
        return _loaded_modules[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from . import analytics, auth, courses, health, hippodromes, reunions
