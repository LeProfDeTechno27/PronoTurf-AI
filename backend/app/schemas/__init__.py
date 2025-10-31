"""Pydantic schemas package with lazy exports."""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, Dict

_module_exports = {
    "user": ("UserCreate", "UserRead", "UserUpdate", "Token"),
    "hippodrome": (
        "HippodromeCreate",
        "HippodromeUpdate",
        "HippodromeResponse",
        "HippodromeSimple",
        "HippodromeList",
    ),
    "reunion": (
        "ReunionCreate",
        "ReunionUpdate",
        "ReunionResponse",
        "ReunionSimple",
        "ReunionWithHippodrome",
        "ReunionList",
        "ReunionDetailResponse",
    ),
    "course": (
        "CourseCreate",
        "CourseUpdate",
        "CourseResponse",
        "CourseSimple",
        "CourseWithReunion",
        "CourseList",
        "CourseDetailResponse",
        "CourseFilter",
    ),
    "horse": (
        "HorseCreate",
        "HorseUpdate",
        "HorseResponse",
        "HorseSimple",
        "HorseList",
        "HorseDetailResponse",
    ),
    "jockey": (
        "JockeyCreate",
        "JockeyUpdate",
        "JockeyResponse",
        "JockeySimple",
        "JockeyList",
        "JockeyDetailResponse",
    ),
    "trainer": (
        "TrainerCreate",
        "TrainerUpdate",
        "TrainerResponse",
        "TrainerSimple",
        "TrainerList",
        "TrainerDetailResponse",
    ),
    "partant": (
        "PartantCreate",
        "PartantUpdate",
        "PartantResponse",
        "PartantSimple",
        "PartantWithRelations",
        "PartantList",
        "PartantDetailResponse",
        "PartantBatchCreate",
        "PartantResultUpdate",
        "PartantOddsUpdate",
    ),
}

_symbol_to_module = {
    symbol: module for module, symbols in _module_exports.items() for symbol in symbols
}

__all__ = list(_symbol_to_module) + list(_module_exports)

_loaded_modules: Dict[str, ModuleType] = {}


def _load_module(module: str) -> ModuleType:
    if module not in _loaded_modules:
        _loaded_modules[module] = import_module(f"{__name__}.{module}")
    return _loaded_modules[module]


def __getattr__(name: str):  # pragma: no cover - passthrough helper
    if name in _module_exports:
        return _load_module(name)
    module_name = _symbol_to_module.get(name)
    if module_name:
        module = _load_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover
    from . import course, hippodrome, horse, jockey, partant, reunion, trainer, user
    from .course import *  # noqa: F401,F403
    from .hippodrome import *  # noqa: F401,F403
    from .horse import *  # noqa: F401,F403
    from .jockey import *  # noqa: F401,F403
    from .partant import *  # noqa: F401,F403
    from .reunion import *  # noqa: F401,F403
    from .trainer import *  # noqa: F401,F403
    from .user import *  # noqa: F401,F403
