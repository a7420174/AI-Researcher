from __future__ import annotations

import os
import json
import time
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class LocalConfig:
    workplace_name: str
    local_root: str = field(default_factory=lambda: os.getcwd())
    conda_path: Optional[str] = None
    python_path: Optional[str] = None
    use_uv: bool = False
    uv_path: Optional[str] = None
    venv_path: Optional[str] = None


class LocalEnv:
    def __init__(self, config: LocalConfig):
        if isinstance(config, dict):
            config = LocalConfig(**config)

        self.config = config
        self.workplace_name = config.workplace_name
        self.local_workplace = str(
            Path(config.local_root).resolve() / config.workplace_name
        )
        self.docker_workplace = f"/{config.workplace_name}"
        self.workplace = self.local_workplace

        self.conda_path = config.conda_path or os.environ.get(
            "CONDA_PATH", "/opt/conda"
        )
        self.python_path = config.python_path or os.environ.get("PYTHON_PATH", "python")
        self.use_uv = (
            config.use_uv or os.environ.get("USE_UV", "false").lower() == "true"
        )
        self.uv_path = config.uv_path or os.environ.get("UV_PATH", "uv")
        self.venv_path = config.venv_path or os.environ.get("VENV_PATH", None)

        self._python_cmd = self._detect_python()

        Path(self.local_workplace).mkdir(parents=True, exist_ok=True)

    def _detect_python(self) -> str:
        if self.use_uv:
            if self._check_command(self.uv_path):
                if self.venv_path and os.path.exists(self.venv_path):
                    return f"{self.uv_path} run --python {self.venv_path}/bin/python"
                return f"{self.uv_path} run python"
            elif self._check_command("uvx"):
                if self.venv_path and os.path.exists(self.venv_path):
                    return f"uvx --python {self.venv_path}/bin/python"
                return "uvx python"

        if self._check_command(self.python_path):
            return self.python_path

        if self._check_command("python3"):
            return "python3"

        activate_conda = f"source {self.conda_path}/etc/profile.d/conda.sh 2>/dev/null || true; conda activate autogpt 2>/dev/null || true"
        return f"{activate_conda} && python"

    def _check_command(self, cmd: str) -> bool:
        try:
            result = subprocess.run(
                ["which", cmd.split()[0]], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    def init_local(self) -> None:
        print(f"[info] Initializing local environment at: {self.local_workplace}")
        if not os.path.exists(self.local_workplace):
            os.makedirs(self.local_workplace, exist_ok=True)

        if self.use_uv and self.venv_path:
            if not os.path.exists(self.venv_path):
                print(f"[info] Creating uv virtual environment at: {self.venv_path}")
                subprocess.run([self.uv_path, "venv", self.venv_path], check=False)

        mode = "uv" if self.use_uv else "conda/system"
        print(f"[ready] Local environment is ready (mode: {mode}).")

    def run_command(
        self,
        command: str,
        stream_callback: Any = None,
        recv_timeout: Optional[float] = 5.0,
    ) -> Dict[str, Any]:
        timeout = int(recv_timeout * 60) if recv_timeout else 300

        if self.use_uv:
            if self._check_command(self.uv_path):
                if self.venv_path and os.path.exists(self.venv_path):
                    full_command = f"cd {self.local_workplace} && {self.uv_path} run --python {self.venv_path}/bin/python {command}"
                else:
                    full_command = f"cd {self.local_workplace} && {self.uv_path} run python {command}"
            else:
                full_command = f"cd {self.local_workplace} && {command}"
        else:
            activate_conda = f"source {self.conda_path}/etc/profile.d/conda.sh 2>/dev/null || true; conda activate autogpt 2>/dev/null || true"
            full_command = f"cd {self.local_workplace} && {activate_conda} && {command}"

        process = None
        try:
            process = subprocess.Popen(
                ["/bin/bash", "-c", full_command],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.local_workplace,
            )

            output_lines = []
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    output_lines.append(line)
                    if stream_callback:
                        stream_callback(line)

            output = "".join(output_lines)
            return_code = process.poll()

            return {
                "status": return_code if return_code is not None else 0,
                "result": output,
            }

        except subprocess.TimeoutExpired:
            if process:
                process.kill()
            return {
                "status": -1,
                "result": f"Command timed out after {timeout} seconds",
            }
        except Exception as e:
            return {"status": -1, "result": f"Error running command: {str(e)}"}

    def run_python(self, code: str, **kwargs) -> Dict[str, Any]:
        return self.run_command(f"python -c {repr(code)}", **kwargs)

    def run_bash(self, command: str, **kwargs) -> Dict[str, Any]:
        return self.run_command(command, **kwargs)
