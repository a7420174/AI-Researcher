# Standard library imports
import asyncio
import copy
import json
import os
import random
import re
import time
from collections import deque, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Union, Any
from zoneinfo import ZoneInfo

# Third-party imports
from httpx import ConnectError, RemoteProtocolError
from litellm import completion, acompletion
from litellm.exceptions import APIError
from litellm import ContextWindowExceededError, BadRequestError
from litellm.types.utils import Message as litellmMessage
from openai import AsyncOpenAI
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Local imports
import litellm
from research_agent.constant import (
    API_BASE_URL,
    MUST_ADD_USER,
    NOT_SUPPORT_FN_CALL,
    NOT_SUPPORT_SENDER,
    NOT_USE_FN_CALL,
)
from research_agent.inno.fn_call_converter import (
    convert_fn_messages_to_non_fn_messages,
    convert_non_fncall_messages_to_fncall_messages,
    convert_tools_to_description,
    interleave_user_into_messages,
    SYSTEM_PROMPT_SUFFIX_TEMPLATE,
)
from research_agent.inno.memory.utils import (
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
)
from .logger import LoggerManager, MetaChainLogger
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessageToolCall,
    Function,
    Message,
    Response,
    Result,
)
from .util import debug_print, function_to_json, merge_chunk, pretty_print_messages

# Constants
DEFAULT_RPM = int(os.getenv("RPM_LIMIT", "30"))
DEFAULT_TPM = int(os.getenv("TPM_LIMIT", "12000"))
SAFE_TPM_RATIO = 0.8
DEFAULT_TPM = int(DEFAULT_TPM * SAFE_TPM_RATIO)
DEFAULT_RPD = int(os.getenv("RPD_LIMIT", "500"))
DEFAULT_MAX_TOKENS = 10000
DEFAULT_ESTIMATED_RESPONSE_TOKENS = 1000
__CTX_VARS_NAME__ = "context_variables"

# litellm.set_verbose = True
litellm.num_retries = 3


def should_retry_error(retry_state: RetryCallState):
    """Check whether the error should be retried."""
    if retry_state.outcome is None:
        return False
    exception = retry_state.outcome.exception()
    if exception is None:
        return False

    print(f"Caught exception: {type(exception).__name__} - {str(exception)}")

    # Match more error types that are usually transient/retriable
    if isinstance(exception, (APIError, RemoteProtocolError, ConnectError)):
        return True
    error_msg = str(exception).lower()
    return any(
        [
            "connection error" in error_msg,
            "server disconnected" in error_msg,
            "eof occurred" in error_msg,
            "timeout" in error_msg,
            "rate limit" in error_msg,
            "rate_limit_error" in error_msg,
            "too many requests" in error_msg,  # HTTP 429
            "resource_exhausted" in error_msg,  # Google/Gemini 계열
            "overloaded" in error_msg,
            "overloaded_error" in error_msg,
            "负载已饱和" in error_msg,
            "error code: 429" in error_msg,
            "context_length_exceeded" in error_msg,
        ]
    )


__CTX_VARS_NAME__ = "context_variables"
logger = LoggerManager.get_logger()


# === [IMPROVED] 토큰 기준 메시지 트렁케이션 ===
def truncate_message(message: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Truncate by token count with a safety cap."""
    if not message:
        return message
    tokens = encode_string_by_tiktoken(message)
    if len(tokens) > max_tokens:
        return decode_tokens_by_tiktoken(tokens[:max_tokens])
    return message


# === [NEW] PT(미국 태평양시) 자정 계산 ===
def next_pt_midnight_utc():
    """Return (now_utc, next_midnight_in_PT_as_UTC)."""
    now_utc = datetime.now(timezone.utc)
    try:
        now_pt = now_utc.astimezone(ZoneInfo("America/Los_Angeles"))
        pt_midnight = now_pt.replace(hour=0, minute=0, second=0, microsecond=0)
        if now_pt >= pt_midnight:
            next_pt_midnight = pt_midnight + timedelta(days=1)
        else:
            next_pt_midnight = pt_midnight
        return now_utc, next_pt_midnight.astimezone(timezone.utc)
    except Exception:
        # Fallback (DST 미고려): -8시간 기준
        pt_now = now_utc - timedelta(hours=8)
        pt_midnight = pt_now.replace(hour=0, minute=0, second=0, microsecond=0)
        if pt_now >= pt_midnight:
            pt_midnight = pt_midnight + timedelta(days=1)
        return now_utc, pt_midnight + timedelta(hours=8)


# === [NEW] 토큰-인식 Rate Limiter (동기/비동기 공통) ===
class TokenAwareRateLimiter:
    """RPM/TPM/RPD 동시 제어. 초과 시 안전 대기."""

    def __init__(
        self,
        rpm: int = DEFAULT_RPM,
        tpm: int = DEFAULT_TPM,
        rpd: int = DEFAULT_RPD,
    ):
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.req_ts: deque[float] = deque()
        self.token_ts: deque[tuple[float, int]] = deque()
        self.daily_count = 0
        self.reset_utc = next_pt_midnight_utc()[1]
        self.response_tokens_history: deque[int] = deque(maxlen=10)
        self._estimated_response_tokens = DEFAULT_ESTIMATED_RESPONSE_TOKENS

    def update_response_tokens(self, actual_tokens: int) -> None:
        """실제 응답 토큰을 기록하고 이동 평균 업데이트."""
        self.response_tokens_history.append(actual_tokens)
        if self.response_tokens_history:
            self._estimated_response_tokens = int(
                sum(self.response_tokens_history) / len(self.response_tokens_history)
            )

    def get_estimated_response_tokens(self) -> int:
        """이동 평균 기반 예측 응답 토큰 반환."""
        return self._estimated_response_tokens

    def _prune(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        while self.req_ts and now - self.req_ts[0] > 60:
            self.req_ts.popleft()
        while self.token_ts and now - self.token_ts[0][0] > 60:
            self.token_ts.popleft()
        if datetime.now(timezone.utc) >= self.reset_utc:
            self.daily_count = 0
            self.reset_utc = next_pt_midnight_utc()[1]

    def _calculate_delay(
        self,
        rpm_ok: bool,
        tpm_ok: bool,
        rpd_ok: bool,
        now: float,
    ) -> float | None:
        delays: list[float] = []
        if not rpm_ok and self.req_ts:
            delays.append(60 - (now - self.req_ts[0]) + 0.05)
        if not tpm_ok and self.token_ts:
            delays.append(60 - (now - self.token_ts[0][0]) + 0.05)
        if not rpd_ok:
            delays.append(
                (self.reset_utc - datetime.now(timezone.utc)).total_seconds() + 1
            )
        if not delays:
            return None
        delay = max(0.5, min(delays))
        return delay * random.uniform(0.8, 1.4)

    def _can_acquire(self, total_est: int) -> bool:
        rpm_ok = len(self.req_ts) < self.rpm
        tpm_used = sum(t for _, t in self.token_ts)
        tpm_ok = (tpm_used + total_est) <= self.tpm
        rpd_ok = self.daily_count < self.rpd
        return rpm_ok and tpm_ok and rpd_ok

    def _record_request(self, total_est: int) -> None:
        now = time.time()
        self.req_ts.append(now)
        self.token_ts.append((now, total_est))
        self.daily_count += 1

    def acquire(self, est_tokens: int) -> None:
        """동기: 필요 시 대기 후 슬롯 확보."""
        est_response = self.get_estimated_response_tokens()
        total_est = est_tokens + est_response
        while True:
            self._prune()
            if self._can_acquire(total_est):
                self._record_request(total_est)
                return
            delay = self._calculate_delay(
                self._can_acquire(total_est),
                self._can_acquire(total_est),
                self._can_acquire(total_est),
                time.time(),
            )
            if delay:
                time.sleep(delay)

    async def acquire_async(self, est_tokens: int) -> None:
        """비동기: 필요 시 대기 후 슬롯 확보."""
        est_response = self.get_estimated_response_tokens()
        total_est = est_tokens + est_response
        while True:
            self._prune()
            if self._can_acquire(total_est):
                self._record_request(total_est)
                return
            delay = self._calculate_delay(
                self._can_acquire(total_est),
                self._can_acquire(total_est),
                self._can_acquire(total_est),
                time.time(),
            )
            if delay:
                await asyncio.sleep(delay)

    async def acquire(self, est_tokens: int) -> None:
        """비동기 acquire (호환성 위한 별칭)."""
        await self.acquire_async(est_tokens)


# === [NEW] 메시지 토큰 추정 ===
def estimate_tokens_from_messages(messages: List[dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(encode_string_by_tiktoken(c))
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text":
                    total += len(encode_string_by_tiktoken(part.get("text", "")))
                else:
                    total += 64
        else:
            total += len(encode_string_by_tiktoken(str(c)))
    return max(1, total)


class MetaChain:
    def __init__(
        self,
        log_path: Union[str, None, MetaChainLogger] = None,
        rpm: int = DEFAULT_RPM,
        tpm: int = DEFAULT_TPM,
        rpd: int = DEFAULT_RPD,
    ):
        """
        log_path: path to the log file; if None, logs will not be saved to disk.
        rpm/tpm/rpd: 레이트리밋 상한 (프로젝트/티어/모델별 실제 값에 맞게 조정)
        """
        if isinstance(log_path, MetaChainLogger):
            self.logger = log_path
        elif log_path:
            self.logger = MetaChainLogger(log_path=log_path)
        else:
            self.logger = logger if logger else LoggerManager.get_logger()

        if self.logger.log_path is None:
            self.logger.info(
                "[Warning] No log path specified, logs will not be saved",
                "...",
                title="Log Path",
                color="light_cyan3",
            )
        else:
            self.logger.info(
                "Log file is saved to",
                self.logger.log_path,
                "...",
                title="Log Path",
                color="light_cyan3",
            )

        # [NEW] 동기/비동기 리미터
        self.rate_limiter = TokenAwareRateLimiter(rpm, tpm, rpd)
        self.rate_limiter_async = TokenAwareRateLimiter(rpm, tpm, rpd)

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Message:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        if agent.examples:
            examples = (
                agent.examples(context_variables)
                if callable(agent.examples)
                else agent.examples
            )
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history
        # debug_print(debug, "Getting chat completion for...:", messages)

        # [NEW] 호출 전 토큰 추정 & 리미터 획득
        est_tokens = estimate_tokens_from_messages(messages)
        self.rate_limiter.acquire(est_tokens)

        tools = [function_to_json(f) for f in agent.functions]
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        create_model = model_override or agent.model

        if any(ns in create_model for ns in NOT_USE_FN_CALL):
            import asyncio
            import copy

            messages_copy = copy.deepcopy(messages)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.get_chat_completion_async(
                            agent,
                            messages_copy,
                            context_variables,
                            model_override,
                            stream,
                            debug,
                        ),
                    )
                    return future.result()
            else:
                return asyncio.run(
                    self.get_chat_completion_async(
                        agent,
                        messages_copy,
                        context_variables,
                        model_override,
                        stream,
                        debug,
                    )
                )

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
            "base_url": API_BASE_URL,
            "max_retries": 3,
        }
        NO_SENDER_MODE = any(ns in create_params["model"] for ns in NOT_SUPPORT_SENDER)
        if NO_SENDER_MODE:
            msgs = create_params["messages"]
            for m in msgs:
                m.pop("sender", None)
            create_params["messages"] = msgs
        if tools and create_params["model"].startswith("gpt"):
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls
        response = completion(**create_params)
        if hasattr(response, "usage") and response.usage:
            self.rate_limiter.update_response_tokens(response.usage.completion_tokens)
        return response

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result
            case Agent() as agent:
                return Result(value=json.dumps({"assistant": agent.name}), agent=agent)
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    self.logger.info(
                        error_message, title="Handle Function Result Error", color="red"
                    )
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
        handle_mm_func: Callable[[], str] = None,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # Handle missing tool case and skip to the next tool
            if name not in function_map:
                self.logger.info(
                    f"Tool {name} not found in function map.",
                    title="Tool Call Error",
                    color="red",
                )
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"[Tool Call Error] Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            if args is None:
                args = {}
            elif isinstance(args, list):
                args = args[0] if args else {}

            # debug_print(
            #     debug, f"Processing tool call: {name} with arguments {args}")
            func = function_map[name]
            # Pass context_variables to agent functions if requested in signature
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            try:
                raw_result = function_map[name](**args)
            except Exception as e:
                # if "case_resolved" in name:
                #     raw_result = function_map[name](tool_call.function.arguments)
                # else:
                self.logger.info(
                    f"[Tool Call Error] The execution of tool {name} failed. Error: {e}",
                    title="Tool Call Error",
                    color="red",
                )
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": f"[Tool Call Error] The execution of tool {name} failed. Error: {e}",
                    }
                )
                continue

            result: Result = self.handle_function_result(raw_result, debug)

            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result.value,
                }
            )
            self.logger.pretty_print_messages(partial_response.messages[-1])
            if result.image:
                assert handle_mm_func, (
                    f"handle_mm_func is not provided, but an image is returned by tool call {name}({tool_call.function.arguments})"
                )
                partial_response.messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": handle_mm_func(
                                    name, tool_call.function.arguments
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{result.image}"
                                },
                            },
                        ],
                    }
                )
            # debug_print(debug, "Tool calling: ", json.dumps(partial_response.messages[-1], indent=4), log_path=log_path, title="Tool Calling", color="green")

            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = True,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info(
            "Receiving the task:",
            history[-1]["content"],
            title="Receive Task",
            color="green",
        )

        while len(history) - init_len < max_turns and active_agent:
            completion_resp = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message: Message = completion_resp.choices[0].message
            message.sender = active_agent.name
            # debug_print(debug, "Received completion:", message.model_dump_json(indent=4), log_path=log_path, title="Received Completion", color="blue")
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))  # Avoid OpenAI types

            if not message.tool_calls or not execute_tools:
                self.logger.info("Ending turn.", title="End Turn", color="red")
                break
            partial_response = self.handle_tool_calls(
                message.tool_calls,
                active_agent.functions,
                context_variables,
                debug,
                handle_mm_func=active_agent.handle_mm_func,
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=2, min=30, max=1200),
        retry=should_retry_error,
        before_sleep=lambda s: print(f"Retrying... (attempt {s.attempt_number})"),
    )
    async def get_chat_completion_async(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> Message:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        if agent.examples:
            examples = (
                agent.examples(context_variables)
                if callable(agent.examples)
                else agent.examples
            )
            history = examples + history

        messages = [{"role": "system", "content": instructions}] + history

        # [NEW] 호출 전 토큰 추정 & 비동기 리미터 획득
        est_tokens = estimate_tokens_from_messages(messages)
        await self.rate_limiter_async.acquire(est_tokens)
        # debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # Hide context_variables from the model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        create_model = model_override or agent.model

        if not any(ns in create_model for ns in NOT_USE_FN_CALL):
            # assert litellm.supports_function_calling(model = create_model) == True, f"Model {create_model} does not support function calling, please set `FN_CALL=False` to use non-function calling mode"
            create_params = {
                "model": create_model,
                "messages": messages,
                "tools": tools or None,
                "tool_choice": agent.tool_choice,
                "stream": stream,
                "base_url": API_BASE_URL,
            }
            NO_SENDER_MODE = any(ns in create_model for ns in NOT_SUPPORT_SENDER)
            if NO_SENDER_MODE:
                msgs = create_params["messages"]
                for m in msgs:
                    m.pop("sender", None)
                create_params["messages"] = msgs
            if tools and create_params["model"].startswith("gpt"):
                create_params["parallel_tool_calls"] = agent.parallel_tool_calls
            completion_response = await acompletion(**create_params)
        else:
            assert agent.tool_choice == "required", (
                f"Non-function calling mode MUST use tool_choice = 'required' rather than {agent.tool_choice}"
            )
            last_content = messages[-1]["content"]
            tools_description = convert_tools_to_description(tools)
            messages[-1]["content"] = (
                last_content
                + "\n[IMPORTANT] You MUST use the tools provided to complete the task.\n"
                + SYSTEM_PROMPT_SUFFIX_TEMPLATE.format(description=tools_description)
            )
            NO_SENDER_MODE = any(ns in create_model for ns in NOT_SUPPORT_SENDER)
            if NO_SENDER_MODE:
                for m in messages:
                    m.pop("sender", None)
            if create_model in NOT_SUPPORT_FN_CALL:
                messages = convert_fn_messages_to_non_fn_messages(messages)
            if create_model in MUST_ADD_USER and messages[-1]["role"] != "user":
                messages = interleave_user_into_messages(messages)
            create_model = "deepseek/deepseek-chat"
            create_params = {
                "model": create_model,
                "messages": messages,
                "stream": stream,
                "base_url": API_BASE_URL,
            }
            completion_response = await acompletion(**create_params)
            last_message = [
                {
                    "role": "assistant",
                    "content": completion_response.choices[0].message.content,
                }
            ]
            converted_message = convert_non_fncall_messages_to_fncall_messages(
                last_message, tools
            )
            converted_tool_calls = [
                ChatCompletionMessageToolCall(**tc)
                for tc in converted_message[0]["tool_calls"]
            ]
            completion_response.choices[0].message = litellmMessage(
                content=converted_message[0]["content"],
                role="assistant",
                tool_calls=converted_tool_calls,
            )
        if hasattr(completion_response, "usage") and completion_response.usage:
            self.rate_limiter_async.update_response_tokens(
                completion_response.usage.completion_tokens
            )
        return completion_response

    async def try_completion_with_truncation(
        self, agent, history, context_variables, model_override, stream, debug
    ):
        try:
            return await self.get_chat_completion_async(
                agent, history, context_variables, model_override, stream, debug
            )
        except (ContextWindowExceededError, BadRequestError) as e:
            msg = str(e).lower()
            if "context length" in msg or "context_length_exceeded" in msg:
                if history and len(history) > 0:
                    last_message = history[-1]
                    if isinstance(last_message.get("content"), str):
                        last_message["content"] = truncate_message(
                            last_message["content"], max_tokens=10000
                        )
                        self.logger.info(
                            "Message has been truncated to fit within context length limits.",
                            title="Message Truncated",
                            color="yellow",
                        )
                        return await self.get_chat_completion_async(
                            agent,
                            history,
                            context_variables,
                            model_override,
                            stream,
                            debug,
                        )
            raise e

    async def run_async(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = True,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        assert stream == False, "Async run does not support stream"
        active_agent = agent
        enter_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        self.logger.info(
            "Receiving the task:",
            history[-1]["content"],
            title="Receive Task",
            color="green",
        )

        while len(history) - init_len < max_turns and active_agent:
            try:
                completion_response = await self.try_completion_with_truncation(
                    agent=active_agent,
                    history=history,
                    context_variables=context_variables,
                    model_override=model_override,
                    stream=stream,
                    debug=debug,
                )
            except Exception as e:
                self.logger.info(f"Error: {e}", title="Error", color="red")
                history.append({"role": "error", "content": f"Error: {e}"})
                break
            message: Message = completion_response.choices[0].message
            message.sender = active_agent.name
            self.logger.pretty_print_messages(message)
            history.append(json.loads(message.model_dump_json()))

            def _is_completion_response(msg: Message) -> bool:
                content = msg.content or ""
                return "<fully_correct>" in content or "<result>" in content

            if enter_agent.tool_choice != "required":
                if (
                    not message.tool_calls and active_agent.name == enter_agent.name
                ) or not execute_tools:
                    self.logger.info("Ending turn.", title="End Turn", color="red")
                    break
            elif message.tool_calls and message.tool_calls[0].function.name in [
                "case_resolved",
                "case_not_resolved",
            ]:
                try:
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                    history.extend(partial_response.messages)
                    context_variables.update(partial_response.context_variables)
                    if not partial_response.messages[-1]["content"].startswith(
                        "[Tool Call Error]"
                    ):
                        self.logger.info(
                            "Ending turn with case resolved/not resolved.",
                            title="End Turn",
                            color="red",
                        )
                        break
                except Exception as e:
                    self.logger.info(f"Error: {e}", title="Error", color="red")
                    history.append({"role": "error", "content": f"Error: {e}"})
                    break
            elif not execute_tools:
                self.logger.info("Ending turn.", title="End Turn", color="red")
                break

            if message.tool_calls:
                try:
                    partial_response = self.handle_tool_calls(
                        message.tool_calls,
                        active_agent.functions,
                        context_variables,
                        debug,
                        handle_mm_func=active_agent.handle_mm_func,
                    )
                except Exception as e:
                    self.logger.info(f"Error: {e}", title="Error", color="red")
                    history.append({"role": "error", "content": f"Error: {e}"})
                    break
            else:
                partial_response = Response(
                    messages=[
                        {
                            "role": "user",
                            "content": "Please use the tools provided to complete the task.",
                        }
                    ]
                )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
