import os
from research_agent.inno.registry import registry
from research_agent.inno.tools import bootstrap_import as bootstrap_tools
from research_agent.inno.agents import bootstrap_import as bootstrap_agents
import research_agent.inno.tools as tools_pkg
import research_agent.inno.agents as agents_pkg

def bootstrap_registry(*, reset_first: bool = False, quiet: bool = True) -> None:
    if reset_first:
        registry.reset()
        
    tools_base_dir  = os.path.dirname(tools_pkg.__file__)
    tools_base_name = tools_pkg.__name__          # "research_agent.inno.tools"
    agents_base_dir = os.path.dirname(agents_pkg.__file__)
    agents_base_name = agents_pkg.__name__        # "research_agent.inno.agents"

    bootstrap_tools(base_dir=tools_base_dir,  base_package=tools_base_name,  quiet=quiet)
    bootstrap_agents(base_dir=agents_base_dir, base_package=agents_base_name, quiet=quiet)

    print("[bootstrap-registry] tools:", ", ".join(sorted(registry.tools.keys())))
    print("[bootstrap-registry] agents:", ", ".join(sorted(registry.agents.keys())))
