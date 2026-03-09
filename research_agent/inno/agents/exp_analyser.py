from __future__ import annotations

from typing import List, Union

from research_agent.inno.types import Agent, Result

# ✅ 개별 툴 함수 import 제거
#    환경 주입 래퍼와 환경 타입만 유지
from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.docker_env import with_env as with_env_docker
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.environment.local_env import LocalEnv
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser

# ✅ registry에서 툴을 이름으로 조회/래핑
from research_agent.inno.registry import get_tools, register_agent


def case_resolved(
    context_variables: dict, analysis_report: str, further_plan: dict[str, str]
):
    """
    After you have carefully and comprehensively reviewed the existing resources and the current project,
    and fully understood the innovative idea, use this function to provide:
      - a detailed analysis report of the existing experiments, and
      - a further plan for the Machine Learning Agent to run additional experiments.

    Args:
        analysis_report (str): The analysis report of existing experiments.
        further_plan (dict[str, str]): The further plan (experiment name -> description).
    """
    if "experiment_report" not in context_variables:
        context_variables["experiment_report"] = []
    context_variables["experiment_report"].append(
        {"analysis_report": analysis_report, "further_plan": further_plan}
    )
    ret_val = f"""\
You have provided the analysis report of existing experiments and a further plan for the `Machine Learning Agent`.
The analysis report is: {analysis_report}
The further plan is: {further_plan}
"""
    return Result(
        value=ret_val,
        context_variables=context_variables,
    )


@register_agent(
    "get_exp_analyser_agent"
)  # 선택: 레지스트리에 팩토리 등록 (이름은 기존 호출부와 일치)
def get_exp_analyser_agent(model: str = "gpt-4o", **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"
    code_env: Union[DockerEnv, LocalEnv] = kwargs.get("code_env", None)
    assert code_env is not None, "code_env is required"

    def instructions(context_variables: dict):
        working_dir = code_env.workplace
        return f"""\
You are given an innovative idea and some experimental results conducted by the `Machine Learning Agent` in `{working_dir}/projects/` to implement the idea.
You also have some reference codebases and papers in the working directory `{working_dir}`.

Your task is to:
1. Analyze the experimental results and provide a detailed analysis report.
2. Analyze the reference codebases and papers, and propose a further plan for additional experiments. The further experiments may include (but are not limited to):
   - Modifying the implementation to better fit the idea.
   - Adding experiments to demonstrate effectiveness and superiority.
   - Visualizing experimental results and providing detailed analysis.
   - Replicating or adapting experiments commonly used in concurrent reference papers and codebases.

AVAILABLE TOOLS:
1. Project and Codebase Navigation:
   - `gen_code_tree_structure`: Understand repository structure
   - `read_file`: Inspect specific implementations
   - `terminal_page_down`, `terminal_page_up`, `terminal_page_to`: Scroll terminal output when it is too long
2. Local File Navigation:
   - `open_local_file`: Open and read paper files
   - `page_up_markdown` / `page_down_markdown`: Navigate pages
   - `find_on_page_ctrl_f` / `find_next`: Search content
   - `visualizer`: SEE experimental results (image/video) and answer a question about them.
     Use this tool for generated images or visualizations to write a thorough analysis.

[IMPORTANT]
Only after you have carefully and comprehensively analyzed the experimental results and the reference codebases/papers,
you should summarize your findings and propose the further plan using the `case_resolved` function.
Do NOT call this function prematurely.
"""

    # ✅ 파일 뷰어/분석 툴: registry에서 이름 기반으로 로드 + file_env 필요 시 자동 주입
    file_tool_names = [
        "open_local_file",
        "page_up_markdown",
        "page_down_markdown",
        "find_on_page_ctrl_f",
        "find_next",
        "question_answer_on_whole_page",
        "visualizer",
    ]
    tool_files = get_tools(file_tool_names, env=file_env, env_wrapper=with_env_file)

    # ✅ 코드/터미널 툴: registry에서 이름 기반으로 로드 + code_env 필요 시 자동 주입
    code_tool_names = [
        "gen_code_tree_structure",
        "read_file",
        "terminal_page_down",
        "terminal_page_up",
        "terminal_page_to",
    ]
    tool_codes = get_tools(code_tool_names, env=code_env, env_wrapper=with_env_docker)

    tools = tool_files + tool_codes + [case_resolved]

    return Agent(
        name="Experiment Analysis Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
