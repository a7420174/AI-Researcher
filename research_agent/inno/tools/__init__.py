# research_agent/inno/tools/__init__.py
import os
import importlib
from typing import Iterable, Optional

from research_agent.inno.registry import registry, get_tool, get_tools


def bootstrap_import(
    modules: Optional[Iterable[str]] = None,
    *,
    base_dir: Optional[str] = None,
    base_package: Optional[str] = None,
    quiet: bool = True,
) -> None:
    """
    Optionally import tool modules to trigger @register_tool decorators.

    - If `modules` is provided: import each module path (preferred for explicitness).
    - Else if base_dir & base_package are provided: recursively import .py files (legacy compatibility).
    """
    if modules:
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception as e:
                if not quiet:
                    print(f"[tools.bootstrap] failed to import {m}: {e}")
        return

    if base_dir and base_package:
        for root, _, files in os.walk(base_dir):
            rel = os.path.relpath(root, base_dir)
            for f in files:
                if f.endswith(".py") and not f.startswith("__"):
                    module = (
                        f"{base_package}.{f[:-3]}"
                        if rel == "."
                        else f"{base_package}.{rel.replace(os.path.sep, '.')}.{f[:-3]}"
                    )
                    try:
                        importlib.import_module(module)
                    except Exception as e:
                        if not quiet:
                            print(f"[tools.bootstrap] failed to import {module}: {e}")


# Re-export registry views
tools = registry.tools
tools_info = registry.tools_info

__all__ = ["bootstrap_import", "tools", "tools_info", "get_tool", "get_tools"]
