# research_agent/inno/agents/__init__.py
import os
import importlib
from typing import Iterable, Optional

from research_agent.inno.registry import (
    registry,
    get_agent_factory,
)

def bootstrap_import(
    modules: Optional[Iterable[str]] = None,
    *,
    base_dir: Optional[str] = None,
    base_package: Optional[str] = None,
    quiet: bool = True,
) -> None:
    """
    Optionally import agent modules to trigger @register_agent decorators.

    Preferred: pass explicit module paths via `modules`.
    Legacy: if `base_dir` & `base_package` are provided, recursively import .py files.
    """
    if modules:
        for m in modules:
            try:
                importlib.import_module(m)
            except Exception as e:
                if not quiet:
                    print(f"[agents.bootstrap] failed to import {m}: {e}")
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
                            print(f"[agents.bootstrap] failed to import {module}: {e}")

# Re-export registry views
agents = registry.agents
agents_info = registry.agents_info

__all__ = ["bootstrap_import", "agents", "agents_info", "get_agent_factory"]