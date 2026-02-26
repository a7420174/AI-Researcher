import json
import os
import hashlib
from typing import Union, Dict, List, Callable, Any, Optional

from research_agent.inno.core import MetaChain, MetaChainLogger


def _hash_content(content: Any) -> str:
    """Generate a short hash from content for cache key."""
    content_str = json.dumps(content, sort_keys=True, default=str)
    return hashlib.sha256(content_str.encode()).hexdigest()[:16]


from research_agent.inno import Agent
from abc import ABC, abstractmethod
from torch import nn  # kept for backward-compat if downstream imports expect it

# --- Optional global confirm hook (provided by Gradio UI layer) ---
try:
    import global_state  # expected to expose CONFIRM_HOOK callable
except Exception:  # pragma: no cover

    class _Dummy:
        CONFIRM_HOOK = None

    global_state = _Dummy()


def _get_cache_policy(default: str = "use") -> str:
    """Resolve cache policy from environment.
    Supported: 'use' | 'rebuild' | 'ask'
    """
    policy = os.getenv("CACHE_POLICY", default).strip().lower()
    return policy if policy in {"use", "rebuild", "ask"} else default


def _normalize_choice(value: Any, *, tri_state: bool = False) -> str:
    """Normalize various return types (bool/str/None) to one of
    'use' | 'resume' | 'rebuild'.

    - For bool: True -> 'use', False -> 'rebuild'
    - For str: map common synonyms
    - tri_state: if False, never returns 'resume'
    """
    if isinstance(value, bool):
        return "use" if value else "rebuild"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"yes", "use", "y", "u", "ok", "accept"}:
            return "use"
        if tri_state and v in {"resume", "continue", "r"}:
            return "resume"
        if v in {"no", "rebuild", "n", "discard", "reset"}:
            return "rebuild"
    # default fallback
    return "use" if not tri_state else "use"


def _confirm(
    prompt: str,
    *,
    choices: Optional[List[str]] = None,
    default: str = "use",
    confirm_fn: Optional[Callable[..., Any]] = None,
    tri_state: bool = False,
) -> str:
    """Ask UI layer for confirmation. Falls back to default when no hook is available.

    Attempt calling with signature (prompt, choices=choices) first; if not supported,
    retry with (prompt) only. Normalize the response to 'use'|'resume'|'rebuild'.
    """
    hook = confirm_fn or getattr(global_state, "CONFIRM_HOOK", None)
    if callable(hook):
        try:
            res = hook(prompt, choices=choices) if choices is not None else hook(prompt)
        except TypeError:
            res = hook(prompt)
        return _normalize_choice(res, tri_state=tri_state)
    return default


class AgentModule:
    def __init__(
        self,
        agent: Agent,
        client: MetaChain,
        cache_path: str,
        *,
        policy: Optional[str] = None,
        confirm_fn: Optional[Callable[..., Any]] = None,
    ):
        self.agent = agent
        self.client = client
        self.cache_path = cache_path
        self.policy = policy  # None -> resolve from env at runtime
        self.confirm_fn = confirm_fn

    async def __call__(
        self,
        messages: List[Dict],
        context_variables: Dict,
        iter_times: int = None,
        *args,
        **kwargs,
    ):
        # messages = [{"role": "user", "content": query}]
        agent_cache, escape_running = self.check_cache(
            self.agent.name, messages, iter_times
        )
        if agent_cache and escape_running:
            messages.extend(
                agent_cache["messages"]
            )  # reuse cached messages, no further run
            context_variables.update(agent_cache["context_variables"])
        elif agent_cache and not escape_running:
            # warm start (resume): extend messages, then run one more step
            messages.extend(agent_cache["messages"])
            context_variables.update(agent_cache["context_variables"])
            response = await self.client.run_async(
                self.agent, messages, context_variables=context_variables, debug=True
            )
            ret_messages = response.messages
            ret_context_variables = response.context_variables
            if ret_messages[-1]["role"] != "error":
                ret_messages.append(
                    {
                        "role": "success",
                        "content": "The agent successfully generated a response.",
                    }
                )
            # save merged cache (previous + new, excluding the final success sentinel)
            self.save_cache(
                self.agent.name,
                agent_cache["messages"] + ret_messages[:-1],
                iter_times,
                ret_context_variables,
            )
            messages.extend(ret_messages[:-1])
            context_variables.update(ret_context_variables)
            if ret_messages[-1]["role"] == "error":
                raise Exception(ret_messages[-1]["content"])
        else:
            # no cache or user chose rebuild
            response = await self.client.run_async(
                self.agent, messages, context_variables=context_variables, debug=True
            )
            ret_messages = response.messages
            ret_context_variables = response.context_variables
            if ret_messages[-1]["role"] != "error":
                ret_messages.append(
                    {
                        "role": "success",
                        "content": "The agent successfully generated a response.",
                    }
                )
            self.save_cache(
                self.agent.name, ret_messages[:-1], iter_times, ret_context_variables
            )
            messages.extend(ret_messages[:-1])
            context_variables.update(ret_context_variables)
            if ret_messages[-1]["role"] == "error":
                raise Exception(ret_messages[-1]["content"])

        return messages, context_variables

    def save_cache(
        self,
        agent_name,
        messages,
        iter_times: int = None,
        context_variables: Dict = None,
    ):
        agent_name = agent_name.replace(" ", "_").lower()
        if iter_times is not None:
            agent_name = agent_name + f"_iter_{iter_times}"
        messages_hash = _hash_content(messages)
        agent_cache_file = f"{self.cache_path}/agents/{agent_name}_{messages_hash}.json"
        os.makedirs(os.path.dirname(agent_cache_file), exist_ok=True)
        with open(agent_cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {"messages": messages, "context_variables": context_variables},
                f,
                ensure_ascii=False,
                indent=4,
            )

    def check_cache(
        self, agent_name, messages: List[Dict] = None, iter_times: int = None
    ):
        agent_name_norm = agent_name.replace(" ", "_").lower()
        if iter_times is not None:
            agent_name_norm = agent_name_norm + f"_iter_{iter_times}"

        if messages is None:
            return None, False

        messages_hash = _hash_content(messages)
        cache_file = f"{self.cache_path}/agents/{agent_name_norm}_{messages_hash}.json"
        if not os.path.exists(cache_file):
            return None, False

        policy = (self.policy or _get_cache_policy()).lower()
        if policy == "use":
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f), True
        if policy == "rebuild":
            return None, False
        choice = _confirm(
            f"The agent '{agent_name}' cache file exists, what do you want to do?",
            choices=["Use", "Resume", "Rebuild"],
            default="use",
            confirm_fn=self.confirm_fn,
            tri_state=True,
        )
        if choice == "use":
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f), True
        if choice == "resume":
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f), False
        return None, False


class ToolModule:
    def __init__(
        self,
        tool: Callable[[Any], Union[str, Dict]],
        cache_path: str,
        *,
        policy: Optional[str] = None,
        confirm_fn: Optional[Callable[..., Any]] = None,
    ):
        self.tool = tool
        self.cache_path = cache_path
        self.policy = policy
        self.confirm_fn = confirm_fn

    def __call__(self, tool_args: Dict, *args, **kwargs):
        tool_cache = self.check_cache(self.tool.__name__, tool_args)
        if tool_cache is not None:
            return tool_cache
        tool_result = self.tool(**tool_args)
        self.save_cache(self.tool, tool_args, tool_result)
        return tool_result

    def save_cache(
        self, tool: Callable, tool_args: Dict, tool_result: Union[str, Dict]
    ):
        tool_name = tool.__name__
        args_hash = _hash_content(tool_args)
        tool_cache_file = f"{self.cache_path}/tools/{tool_name}_{args_hash}.json"
        os.makedirs(os.path.dirname(tool_cache_file), exist_ok=True)
        tool_cache_dict = {"name": tool_name, "args": tool_args, "result": tool_result}
        with open(tool_cache_file, "w", encoding="utf-8") as f:
            json.dump(tool_cache_dict, f, ensure_ascii=False, indent=4)

    def check_cache(self, tool_name: str, tool_args: Dict = None):
        if tool_args is None:
            return None
        args_hash = _hash_content(tool_args)
        cache_file = f"{self.cache_path}/tools/{tool_name}_{args_hash}.json"
        if not os.path.exists(cache_file):
            return None

        policy = (self.policy or _get_cache_policy()).lower()
        if policy == "use":
            with open(cache_file, "r", encoding="utf-8") as f:
                tool_cache_dict = json.load(f)
                return tool_cache_dict["result"]
        if policy == "rebuild":
            return None
        choice = _confirm(
            f"The tool '{tool_name}' cache file exists, do you want to use it?",
            choices=["Yes", "No"],
            default="use",
            confirm_fn=self.confirm_fn,
            tri_state=False,
        )
        if choice == "use":
            with open(cache_file, "r", encoding="utf-8") as f:
                tool_cache_dict = json.load(f)
                return tool_cache_dict["result"]
        return None


class FlowModule(ABC):
    def __init__(
        self,
        cache_path: str,
        log_path: Union[str, None, MetaChainLogger] = None,
        model: str = "gpt-4o-2024-08-06",
        *,
        policy: Optional[str] = None,
        confirm_fn: Optional[Callable[..., Any]] = None,
    ):
        self.cache_path = cache_path
        self.client = MetaChain(log_path=log_path)
        self.model = model
        self.policy = policy
        self.confirm_fn = confirm_fn

    @abstractmethod
    async def forward(self, *args, **kwargs):
        raise NotImplementedError("subclass should implement this method")

    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)
