from __future__ import annotations

from importlib import import_module
from types import ModuleType

_MODULE_ORDER = ("datagen", "dataloader")
_OPTIONAL_DATALOADER_IMPORTS = {"torch", "zarr"}


def _maybe_import_data_module(module_name: str) -> ModuleType | None:
    try:
        return import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        if module_name == "dataloader" and exc.name in _OPTIONAL_DATALOADER_IMPORTS:
            return None
        raise


def __getattr__(name: str):
    for module_name in _MODULE_ORDER:
        module = _maybe_import_data_module(module_name)
        if module is not None and hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _MODULE_ORDER:
        module = _maybe_import_data_module(module_name)
        if module is None:
            continue
        names.update(k for k in module.__dict__ if not k.startswith("_"))
    return sorted(names)
