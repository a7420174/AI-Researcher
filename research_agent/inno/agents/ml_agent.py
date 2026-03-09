from typing import Dict, Any, Union
import json

from research_agent.inno.types import Agent
from research_agent.inno.registry import (
    register_agent,
    get_tools,
)  # ← 레지스트리 API만 사용
from research_agent.inno.environment.docker_env import (
    DockerEnv,
    with_env as with_env_docker,
)
from research_agent.inno.environment.local_env import LocalEnv


def case_resolved(task_response):
    """
    The task response is the result of the task. Use this function only after you have successfully completed the task.

    Args:
        task_response: The result of the task.
    """
    return task_response


def case_not_resolved(failure_reason):
    """
    The failure reason is the reason why you cannot find a solution to the task.s
    Use this function only after you have tried multiple times and still cannot find a solution.

    Args:
       failure_reason: The reason why you cannot find a solution to the task.
    """
    return failure_reason


@register_agent("get_ml_agent")
def get_ml_agent(model: str, **kwargs):
    code_env: Union[DockerEnv, LocalEnv] = kwargs.get("code_env", None)

    def instructions(context_variables):
        working_dir = code_env.workplace
        return f"""\
You are a machine learning engineer tasked with implementing innovative ML projects. Your workspace is: `{working_dir}`.

OBJECTIVE:
Create a self-contained, well-organized implementation in `{working_dir}/project` based on:
- The provided innovative idea
- Reference codebases (up to 5 repositories)
- The detailed implementation plan

CODE INTEGRATION PRINCIPLES:
1. Self-Contained Project
   - ALL code must reside within the project directory
   - NO direct imports from reference codebases
   - Reference code must be thoughtfully integrated into your project structure
   - Maintain consistent coding style across integrated components

2. Code Adaptation Guidelines
   - Study reference implementations thoroughly
   - Understand the core logic and algorithms
   - Rewrite and adapt code to fit your project's architecture
   - Document the origin and modifications of adapted code
   - Ensure consistent naming conventions and style

AVAILABLE TOOLS:
1. Project Structure:
   - `create_directory`: Create organized project structure
   - `create_file`, `write_file`: Write clean, documented code
   - `list_files`, `read_file`: Examine existing code
   - `terminal_page_down`, `terminal_page_up` and `terminal_page_to`: Scroll the terminal output when it is too long. You can use `terminal_page_to` to move the viewport to the specific page of terminal where the meaningful content is, for example, when the terminal output contains a progress bar or output of generating directory structure when there are many datasets in the directory, you can use `terminal_page_to` to move the viewport to the end of terminal where the meaningful content is.
2. Execution:
   - `run_python`: Run scripts without arguments
   - `execute_command`: Run with environment variables/arguments
   Note: When using `execute_command`, use `cd xx` instead of `cwd=xx`

IMPORTANT NOTES:
1. Code Integration
   - DO NOT import directly from reference codebases
   - DO adapt and integrate code thoughtfully
   - DO document code origins and modifications

2. Project Independence
   - Ensure all dependencies are explicitly declared
   - Include all necessary utility functions
   - Maintain clean separation from reference code
   - Create a truly self-contained project

3. Implementation Checklist
   - Verify each model component against the plan
   - Confirm dataset matches specifications
   - Document any deviations or modifications
   - NO shortcuts or simplifications without approval

Remember: Your goal is to create a well-organized, self-contained project that:
1. Implements EVERY component from the model plan exactly as specified
2. Uses the EXACT datasets from the plan (no toy data)
3. Thoughtfully incorporates ideas from reference implementations
4. Maintains its own coherent structure
5. You should integrate ALL academic definition and their code implementation into the project.
"""

    # 레지스트리에서 이름으로 도구를 일괄 조회하고,
    # 툴 시그니처에 'env'가 있을 때만 with_env_docker로 자동 주입됩니다.
    tool_names = [
        "gen_code_tree_structure",
        "execute_command",
        "read_file",
        "create_file",
        "write_file",
        "list_files",
        "create_directory",
        "run_python",
        "terminal_page_down",
        "terminal_page_up",
        "terminal_page_to",
    ]
    tools = get_tools(
        tool_names, env=code_env, env_wrapper=with_env_docker
    )  # ← 안전 조회/주입 [1](https://amcsciences-my.sharepoint.com/personal/a7420174_amcsciences_com/Documents/Microsoft%20Copilot%20Chat%20%ED%8C%8C%EC%9D%BC/registry.py)

    # 레지스트리에 없는 로컬 함수(case_resolved/not_resolved)는 직접 추가
    tools.extend([case_resolved, case_not_resolved])

    return Agent(
        name="Machine Learning Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
