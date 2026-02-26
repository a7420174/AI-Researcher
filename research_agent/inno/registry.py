from __future__ import annotations

import inspect
import threading
from typing import Callable, Dict, Literal, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class FunctionInfo:
    """
    등록된 함수(툴/에이전트)에 대한 메타정보 컨테이너.
    - func: 실제 callable (직렬화 불가 → to_dict()에서 제외)
    - args: 파라미터 이름 리스트
    - docstring: 함수의 문서 문자열
    - body: 함수 본문 소스(데코레이터/def 시그니처 제외, getsource 실패 시 빈 문자열)
    - return_type: 반환 타입힌트 문자열(없으면 None)
    """
    name: str
    func: Callable
    args: List[str]
    docstring: Optional[str]
    body: str
    return_type: Optional[str]

    def to_dict(self) -> dict:
        """직렬화 가능한 dict로 변환 (실행 객체인 func는 제외)."""
        d = asdict(self)
        d.pop('func', None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FunctionInfo":
        """
        dict로부터 FunctionInfo 복원. (실행은 불가하므로 func=None으로 둠)
        """
        if 'func' not in data:
            data['func'] = None
        return cls(**data)


class Registry:
    """
    툴/에이전트를 단일 레지스트리에서 관리하는 싱글톤.
    - register(): 데코레이터로 등록
    - tools / agents: 이름 → callable 매핑
    - tools_info / agents_info: 이름 → FunctionInfo 매핑
    - get_tool / get_agent_factory / get_tools: 안전한 조회(가용 키 안내)
    """
    _instance = None

    # 내부 저장소
    _registry: Dict[str, Dict[str, Callable]] = {"tools": {}, "agents": {}}
    _registry_info: Dict[str, Dict[str, FunctionInfo]] = {"tools": {}, "agents": {}}

    # 동시성 보호용 락
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def reset(self) -> None:
        """레지스트리를 초기화(노트북/핫리로드 환경에서 유용)."""
        with self._lock:
            self._registry = {"tools": {}, "agents": {}}
            self._registry_info = {"tools": {}, "agents": {}}

    def register(
        self,
        type: Literal["tool", "agent"],
        name: str | None = None,
        *,
        allow_override: bool = False
    ):
        """
        통합 등록 데코레이터.

        Args:
            type: "tool" 또는 "agent"
            name: 등록 이름(미지정 시 함수명)
            allow_override: 동일 이름 존재 시 덮어쓸지 여부 (기본 False → 예외)
        """
        def decorator(func: Callable):
            nonlocal name
            reg_name = name or func.__name__

            # 함수 시그니처/문서
            signature = inspect.signature(func)
            args = list(signature.parameters.keys())
            docstring = inspect.getdoc(func)

            # 함수 본문(인터랙티브/빌트인 등 getsource 실패 폴백)
            try:
                source_lines = inspect.getsource(func)
                # 데코레이터/def 라인을 건너뛰고 본문만 추출
                body_lines = source_lines.split('\n')[1:]
                while body_lines and (body_lines[0].strip().startswith('@') or 'def ' in body_lines[0]):
                    body_lines = body_lines[1:]
                body = '\n'.join(body_lines)
            except (OSError, TypeError):
                body = ""

            # 반환 타입힌트
            return_type = None
            if signature.return_annotation != inspect.Signature.empty:
                return_type = str(signature.return_annotation)

            func_info = FunctionInfo(
                name=reg_name,
                func=func,
                args=args,
                docstring=docstring,
                body=body,
                return_type=return_type
            )

            registry_type = f"{type}s"
            with self._lock:
                if not allow_override and reg_name in self._registry[registry_type]:
                    raise ValueError(
                        f"[registry] {type} '{reg_name}'는 이미 등록되어 있습니다. "
                        f"(allow_override=True로 덮어쓰기를 허용할 수 있습니다)"
                    )
                self._registry[registry_type][reg_name] = func
                self._registry_info[registry_type][reg_name] = func_info
            return func
        return decorator

    # -------- 조회용 프로퍼티 --------
    @property
    def tools(self) -> Dict[str, Callable]:
        return self._registry["tools"]

    @property
    def agents(self) -> Dict[str, Callable]:
        return self._registry["agents"]

    @property
    def tools_info(self) -> Dict[str, FunctionInfo]:
        return self._registry_info["tools"]

    @property
    def agents_info(self) -> Dict[str, FunctionInfo]:
        return self._registry_info["agents"]

    # -------- 안전 조회 헬퍼 --------
    def get_tool(self, name: str) -> Callable:
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self.tools.keys()))
            raise KeyError(f"[registry] 등록되지 않은 tool: '{name}'. 사용 가능: [{available}]")
        return tool

    def get_agent_factory(self, name: str) -> Callable:
        fn = self.agents.get(name)
        if fn is None:
            available = ", ".join(sorted(self.agents.keys()))
            raise KeyError(f"[registry] 등록되지 않은 agent factory: '{name}'. 사용 가능: [{available}]")
        return fn

    def get_tools(
        self,
        names: List[str],
        *,
        env: Optional[object] = None,
        env_wrapper: Optional[Callable] = None
    ) -> List[Callable]:
        """
        여러 개의 툴을 이름으로 조회하고, 필요 시 env를 주입해 반환.
        - 툴 시그니처에 'env' 파라미터가 있을 때만 env_wrapper로 래핑합니다.

        Args:
            names: 조회할 툴 이름 리스트
            env: 환경 객체(있으면 with_env로 주입)
            env_wrapper: with_env 함수 (예: with_env_docker, with_env_file)

        Returns:
            List[Callable]: (필요 시 래핑된) 툴 리스트
        """
        out: List[Callable] = []
        for name in names:
            tool = self.tools.get(name)
            if tool is None:
                available = ", ".join(sorted(self.tools.keys()))
                raise KeyError(f"[registry] 등록되지 않은 tool: '{name}'. 사용 가능: [{available}]")
            if env is not None and env_wrapper is not None:
                try:
                    if "env" in inspect.signature(tool).parameters:
                        tool = env_wrapper(env)(tool)
                except (ValueError, TypeError):
                    # 일부 built-in/특수 callable은 signature를 가질 수 없음 → 그대로 사용
                    pass
            out.append(tool)
        return out


# 전역 인스턴스
registry = Registry()


# 편의 데코레이터
def register_tool(name: str | None = None, *, allow_override: bool = False):
    return registry.register(type="tool", name=name, allow_override=allow_override)

def register_agent(name: str | None = None, *, allow_override: bool = False):
    return registry.register(type="agent", name=name, allow_override=allow_override)


# 외부에서 사용하기 쉬운 단일 함수 버전 (기존 코드 호환)
def get_tool(name: str) -> Callable:
    return registry.get_tool(name)

def get_agent_factory(name: str) -> Callable:
    return registry.get_agent_factory(name)

def get_tools(
    names: List[str],
    env: Optional[object] = None,
    env_wrapper: Optional[Callable] = None
) -> List[Callable]:
    return registry.get_tools(names, env=env, env_wrapper=env_wrapper)
