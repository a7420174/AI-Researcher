from __future__ import annotations

import os
import json
import time
import socket
import tarfile
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Callable, Any, Tuple, List
from functools import update_wrapper
from inspect import signature


# --- 안전한 로깅을 위한 마스킹 ---
def _mask_token(s: str) -> str:
    if not s:
        return s
    # 너무 공격적으로 마스킹하지 않고, github 토큰 형태만 간단히 마스킹
    return (
        s.replace(os.environ.get("GITHUB_AI_TOKEN", ""), "***")
        if "GITHUB" in os.environ
        else s
    )


# --- 예외 타입 ---
class DockerEnvError(Exception): ...


class DockerRunError(DockerEnvError): ...


class DockerTimeoutError(DockerEnvError): ...


class GitError(DockerEnvError): ...


class NetworkError(DockerEnvError): ...


class PackageError(DockerEnvError): ...


# --- 안전한 tar 추출 ---
def _safe_tar_extract(tar_path: Path, dest_dir: Path) -> None:
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:

            def is_within_directory(directory, target):
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
                return os.path.commonpath([abs_directory]) == os.path.commonpath(
                    [abs_directory, abs_target]
                )

            for member in tar.getmembers():
                target_path = dest_dir / member.name
                if not is_within_directory(dest_dir, target_path):
                    raise PackageError(f"Blocked path traversal in tar: {member.name}")
            tar.extractall(dest_dir)
    except Exception as e:
        raise PackageError(f"Failed to extract package '{tar_path}': {e}")


# --- 공통 subprocess 실행기 ---
def _run(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    timeout: Optional[int] = 120,
    check: bool = False,
) -> subprocess.CompletedProcess:
    # print로 남길 때는 마스킹
    printable = " ".join(_mask_token(c) for c in cmd)
    # print(f"Running: {printable}")  # 필요 시 디버그
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
        return cp
    except subprocess.TimeoutExpired as e:
        raise DockerTimeoutError(f"Command timed out: {printable}") from e
    except subprocess.CalledProcessError as e:
        # check=True 일 때만 여기로 옴
        raise DockerRunError(
            f"Command failed: {printable}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"
        ) from e
    except Exception as e:
        raise DockerEnvError(f"Command error: {printable} -> {e}") from e


# --- 포트 유틸 ---
def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


def _find_free_port(preferred: Optional[int] = None) -> int:
    if preferred and not _is_port_open("127.0.0.1", preferred):
        return preferred
    # 커널에게 임시 포트 할당 요청
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- Docker 상태 확인 ---
def docker_container_exists(name: str) -> bool:
    cp = _run(
        ["docker", "ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}"]
    )
    return name in cp.stdout.strip().splitlines()


def docker_container_running(name: str) -> bool:
    cp = _run(["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"])
    return name in cp.stdout.strip().splitlines()


def docker_container_inspect(name: str) -> dict:
    cp = _run(["docker", "inspect", name], check=False)
    try:
        arr = json.loads(cp.stdout)
        return arr[0] if arr else {}
    except json.JSONDecodeError:
        return {}


def docker_mapped_host_port(name: str, container_port: int) -> Optional[int]:
    """
    docker port <name> <container_port> 형식 우선 사용. (예: '0.0.0.0:12345')
    """
    cp = _run(["docker", "port", name, str(container_port)], check=False)
    line = cp.stdout.strip()
    if not line:
        return None
    # 형태: 0.0.0.0:12345 / :::12345
    parts = line.split()
    candidate = parts[-1] if parts else line
    if ":" in candidate:
        try:
            return int(candidate.split(":")[-1])
        except ValueError:
            return None
    return None


@dataclass
class DockerConfig:
    container_name: str
    workplace_name: str
    communication_port: int  # host port to map to container_port (default 8000)
    test_pull_name: str = "main"
    task_name: Optional[str] = None
    git_clone: bool = False
    setup_package: Optional[str] = None  # packages/foo.tar.gz의 'foo' 이름
    local_root: str = field(default_factory=lambda: os.getcwd())
    # 선택 필드 (기존 상수 대체 가능)
    base_image: Optional[str] = None  # 없으면 환경변수 BASE_IMAGES 시도
    gpus: Optional[str] = None  # "all", "device=0", "" 등
    platform: Optional[str] = None  # "linux/amd64" 등
    # 네트워크/대기 설정
    container_port: int = 8000  # 컨테이너 내부 TCP 서버 포트
    wait_timeout: int = 90  # 컨테이너 기동/포트 오픈 대기 초
    wait_interval: float = 1.0  # 폴링 간격
    # Git 정보
    git_owner_repo: str = "tjb-tech/metachain"
    ai_user: Optional[str] = None  # 환경변수 AI_USER 또는 전달값
    github_ai_token: Optional[str] = None  # 환경변수 GITHUB_AI_TOKEN 또는 전달값


class DockerEnv:
    def __init__(self, config: Union[DockerConfig, Dict[str, Any]]):
        if isinstance(config, dict):
            config = DockerConfig(**config)

        # 환경변수 fallback
        self.config = config
        self.container_name = config.container_name
        self.workplace_name = config.workplace_name
        self.local_workplace = str(
            Path(config.local_root).resolve() / config.workplace_name
        )
        self.docker_workplace = f"/{config.workplace_name}"
        self.workplace = self.docker_workplace

        self.communication_port = config.communication_port
        self.container_port = config.container_port

        self.test_pull_name = config.test_pull_name
        self.task_name = config.task_name
        self.git_clone = config.git_clone
        self.setup_package = config.setup_package

        self.base_image = config.base_image or os.environ.get("BASE_IMAGES")
        if not self.base_image:
            raise DockerEnvError("base_image(또는 환경변수 BASE_IMAGES)가 필요합니다.")

        self.gpus = (
            config.gpus if config.gpus is not None else os.environ.get("GPUS", "")
        ).strip()
        self.platform = config.platform or os.environ.get("PLATFORM")

        self.ai_user = config.ai_user or os.environ.get("AI_USER", "")
        self.github_ai_token = config.github_ai_token or os.environ.get(
            "GITHUB_AI_TOKEN", ""
        )

        # 작업 디렉토리 준비
        Path(self.local_workplace).mkdir(parents=True, exist_ok=True)

    # ---------------------- Public API ----------------------

    def init_container(self) -> None:
        """
        1) (옵션) 패키지 압축 해제
        2) (옵션) Git clone + 브랜치 생성
        3) 컨테이너 존재/실행 상태 점검
            - 실행 중이면 그대로 사용
            - 존재하나 정지 상태면 start
            - 없으면 run
        4) readiness: docker inspect 상태 + host 포트 오픈 대기
        """
        # 1) 패키지 설치
        if self.setup_package:
            tar_path = Path("packages") / f"{self.setup_package}.tar.gz"
            if not tar_path.exists():
                raise PackageError(f"Package not found: {tar_path}")
            _safe_tar_extract(tar_path, Path(self.local_workplace))

        # 2) git clone
        if self.git_clone:
            self._ensure_metachain_repo()

        # 3) 컨테이너 존재/실행
        if docker_container_exists(self.container_name):
            if docker_container_running(self.container_name):
                # 이미 실행 중 → 포트 동기화
                mapped = docker_mapped_host_port(
                    self.container_name, self.container_port
                )
                if mapped:
                    self.communication_port = mapped
                print(
                    f"[info] Container '{self.container_name}' already running on host:{self.communication_port}"
                )
            else:
                # 정지 상태 → 시작
                self._start_container()
        else:
            # 신규 실행
            self._run_container()

        # 4) readiness 대기: Running + host 포트 오픈
        self._wait_for_ready(
            timeout=self.config.wait_timeout, interval=self.config.wait_interval
        )
        print(
            f"[ready] Container '{self.container_name}' is ready (host:{self.communication_port} -> container:{self.container_port})."
        )

    def stop_container(self) -> None:
        if not docker_container_exists(self.container_name):
            return
        cp = _run(["docker", "stop", self.container_name], check=False)
        if cp.returncode != 0:
            raise DockerRunError(f"Failed to stop container: {cp.stderr}")

    def run_command(
        self,
        command: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        recv_timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        """
        컨테이너 내 TCP 서버(컨테이너 포트: self.container_port, 호스트 포트: self.communication_port)로
        NDJSON 스트리밍 프로토콜(한 줄당 1 JSON 오브젝트)로 명령 전달 및 수신.

        프로토콜:
          - 보냄: UTF-8 텍스트 커맨드 (개행 없이)
          - 받음: {"type":"chunk","data":"..."}* + {"type":"final","status":int,"result":str}

        반환:
          {"status": int, "result": str}
        """
        host, port = "127.0.0.1", int(self.communication_port)
        buffer_size = 8192

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            try:
                s.connect((host, port))
            except Exception as e:
                raise NetworkError(f"Could not connect to {host}:{port} -> {e}") from e

            # 송신
            s.sendall(command.encode("utf-8"))

            # 수신 루프
            s.settimeout(recv_timeout or 5.0)
            partial = ""
            start = time.time()
            while True:
                if recv_timeout and (time.time() - start) > (
                    recv_timeout * 60
                ):  # 안전 상한 (옵션)
                    raise NetworkError(
                        "Timed out waiting for final response from TCP server."
                    )

                try:
                    chunk = s.recv(buffer_size)
                    if not chunk:
                        break
                    data = partial + chunk.decode("utf-8", errors="replace")
                    lines = data.split("\n")
                    for line in lines[:-1]:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            # NDJSON이 아니거나 파편 → 무시/로그
                            # print(f"[warn] Invalid JSON line: {line[:200]}")
                            continue

                        t = msg.get("type")
                        if t == "chunk":
                            if stream_callback:
                                stream_callback(str(msg.get("data", "")))
                        elif t == "final":
                            return {
                                "status": int(msg.get("status", -1)),
                                "result": str(msg.get("result", "")),
                            }
                        else:
                            # 알 수 없는 타입 → 무시
                            pass
                    partial = lines[-1]
                except socket.timeout:
                    # 계속 대기 (서버 스트리밍 지연 허용)
                    continue

            # 연결 종료되었는데 final을 못 받음
            return {"status": -1, "result": "Connection closed without final response"}

    # ---------------------- 내부 메서드 ----------------------

    def _ensure_metachain_repo(self) -> None:
        repo_dir = Path(self.local_workplace) / "metachain"
        if not repo_dir.exists():
            # 안전한 clone (토큰이 있으면 URL에 포함 가능하나, 로그 마스킹 필수)
            owner_repo = self.config.git_owner_repo
            if self.ai_user and self.github_ai_token:
                safe_url = f"https://{self.ai_user}:{self.github_ai_token}@github.com/{owner_repo}.git"
            else:
                # 퍼블릭 또는 로컬 인증(JSON credential helper 등) 기대
                safe_url = f"https://github.com/{owner_repo}.git"

            cp = _run(
                ["git", "clone", "-b", self.test_pull_name, safe_url, str(repo_dir)],
                check=False,
            )
            if cp.returncode != 0:
                raise GitError(
                    f"Failed to clone: {_mask_token(cp.stderr or cp.stdout)}"
                )

        # 브랜치 생성/전환
        new_branch = (
            f"{self.test_pull_name}_{self.task_name}"
            if self.task_name
            else self.test_pull_name
        )
        # 존재 여부 확인
        cp = _run(
            ["git", "rev-parse", "--verify", new_branch], cwd=repo_dir, check=False
        )
        if cp.returncode != 0:
            # 새 브랜치 생성
            cp2 = _run(["git", "checkout", "-b", new_branch], cwd=repo_dir, check=False)
            if cp2.returncode != 0:
                # 이미 있거나 생성 실패 → 스위치 시도
                cp3 = _run(["git", "checkout", new_branch], cwd=repo_dir, check=False)
                if cp3.returncode != 0:
                    raise GitError(
                        f"Failed to create/switch branch '{new_branch}': {cp2.stderr or cp3.stderr}"
                    )
        else:
            # 존재 → 체크아웃
            cp4 = _run(["git", "checkout", new_branch], cwd=repo_dir, check=False)
            if cp4.returncode != 0:
                raise GitError(
                    f"Failed to switch to existing branch '{new_branch}': {cp4.stderr}"
                )

    def _start_container(self) -> None:
        cp = _run(["docker", "start", self.container_name], check=False)
        if cp.returncode != 0:
            raise DockerRunError(
                f"Failed to start container '{self.container_name}': {cp.stderr}"
            )
        mapped = docker_mapped_host_port(self.container_name, self.container_port)
        if mapped:
            self.communication_port = mapped

    def _run_container(self) -> None:
        # 사전 포트 충돌 방지: port 점검 후 가용 포트 선택
        host_port = _find_free_port(self.communication_port)
        self.communication_port = host_port

        cmd = ["docker", "run", "-d"]
        if self.platform:
            cmd += ["--platform", self.platform]
        # userns는 환경에 따라 지원 안될 수 있어 옵션화
        cmd += ["--userns=host"]
        if self.gpus:
            cmd += ["--gpus", self.gpus]
        cmd += [
            "--name",
            self.container_name,
            "--user",
            "root",
            "-v",
            f"{self.local_workplace}:{self.docker_workplace}",
            "-w",
            self.docker_workplace,
            "-p",
            f"{self.communication_port}:{self.container_port}",
            "--restart",
            "unless-stopped",
            self.base_image,
        ]
        cp = _run(cmd, check=False)
        if cp.returncode != 0:
            raise DockerRunError(f"Failed to run container: {cp.stderr or cp.stdout}")

    def _wait_for_ready(self, timeout: int = 60, interval: float = 1.0) -> None:
        start = time.time()
        # 1) Running 대기
        while time.time() - start < timeout:
            if docker_container_running(self.container_name):
                break
            time.sleep(interval)
        else:
            raise DockerTimeoutError(
                f"Container '{self.container_name}' did not enter Running within {timeout}s"
            )

        # 2) 포트 오픈 대기 (호스트 측)
        while time.time() - start < timeout:
            mapped = docker_mapped_host_port(self.container_name, self.container_port)
            if mapped:
                self.communication_port = mapped  # 동기화
            if _is_port_open("127.0.0.1", int(self.communication_port)):
                return
            time.sleep(interval)
        raise DockerTimeoutError(
            f"Port {self.communication_port} not open within {timeout}s"
        )


# ---------------------- 데코레이터 ----------------------


def with_env(env: DockerEnv):
    """
    주어진 env를 kwarg로 주입하는 데코레이터.
    - 원 함수의 시그니처에서 'env' 매개변수를 숨겨 사용자 경험 개선
    - docstring 내 {docker_workplace}, {local_workplace} 치환(안전 처리)
    """

    def decorator(func: Callable[..., Any]):
        def wrapped(*args, **kwargs):
            kwargs["env"] = env
            return func(*args, **kwargs)

        update_wrapper(wrapped, func)
        try:
            params = [p for p in signature(func).parameters.values() if p.name != "env"]
            wrapped.__signature__ = signature(func).replace(parameters=params)
        except Exception:
            # 시그니처 조정 실패 시 무시
            pass

        doc = func.__doc__ or ""
        # 안전한 치환 (중복 포맷 호출 방지)
        doc = doc.replace("{docker_workplace}", env.docker_workplace)
        doc = doc.replace("{local_workplace}", env.local_workplace)
        wrapped.__doc__ = doc
        return wrapped

    return decorator


# ---------------------- 호환 헬퍼 (이름 유지) ----------------------


def check_container_ports(container_name: str) -> Optional[Tuple[int, int]]:
    """
    컨테이너 포트 매핑 확인
    반환 (host_port, container_port) 또는 None
    구현: 'docker port <name>' 파싱
    """
    # 여러 포트가 있을 수 있으나, 여기서는 첫 항목만 선택
    cp = _run(["docker", "port", container_name], check=False)
    out = cp.stdout.strip()
    if not out:
        return None
    # 예:
    # 8000/tcp -> 0.0.0.0:12345
    # 22/tcp   -> 0.0.0.0:2222
    for line in out.splitlines():
        line = line.strip()
        if "->" not in line:
            continue
        left, right = [t.strip() for t in line.split("->", 1)]
        # left: "8000/tcp", right: "0.0.0.0:12345" 혹은 "::1:12345"
        try:
            container_port = int(left.split("/")[0])
        except ValueError:
            continue
        try:
            host_port = int(right.split(":")[-1])
        except ValueError:
            continue
        return (host_port, container_port)
    return None


def check_container_exist(container_name: str) -> bool:
    return docker_container_exists(container_name)


def check_container_running(container_name: str) -> bool:
    return docker_container_running(container_name)
