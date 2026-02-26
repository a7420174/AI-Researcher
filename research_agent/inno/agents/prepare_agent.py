from typing import List
import json

from research_agent.inno.types import Agent, Result
from research_agent.inno.registry import register_agent, get_tools  # ← 레지스트리 API 사용
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.environment.docker_env import with_env as with_env_docker  # ← env 주입 래퍼


def case_resolved(reference_codebases: List[str],
                  reference_paths: List[str],
                  reference_papers: List[str]):
    """
    The function to output the determined reference codebases. Use this function only after you have
    carefully reviewed the existing resources and understand the task.

    Args:
        reference_codebases: list of names of the determined reference codebases.
        reference_paths: list of the determined reference paths.
        reference_papers: list of titles of the determined reference papers.
    """
    prepare_result = {
        "reference_codebases": reference_codebases,
        "reference_paths": reference_paths,
        "reference_papers": reference_papers
    }

    return Result(
        value=f"""\
I have determined the reference codebases and paths according to the existing resources and the innovative ideas.
{json.dumps(prepare_result, ensure_ascii=False, indent=4)}
""",
        context_variables={"prepare_result": prepare_result}
    )


@register_agent("get_prepare_agent")
def get_prepare_agent(model: str, **kwargs):
    code_env: DockerEnv = kwargs.get("code_env", None)

    def instructions(context_variables):
        working_dir = context_variables.get("working_dir", None)
        return f"""
You are given a list of papers, searching results of the papers on GitHub, and innovative ideas according to the papers. Your working directory is `/{working_dir}`, you can only access files in this directory.

Your task is to go through the searching results, find out more detailed information about repositories in the searching results, and determine which repositories are the most relevant and useful to the innovative ideas. You can determine the relevance and usefulness by the following criteria:
1. Repositories with more stars are more recommended.
2. Repositories created more recently are more recommended, [IMPORTANT!] Too old repositories are not recommended.
3. More detaild `README.md` file means more readable codebase and more reproducible, so more recommended.
4. More clear code structure, code comments, and inline code explanations mean more readable codebase and more maintainable, so more recommended.
5. I prefer repositories with `python` language, and running coding in the local machine rather than in docker. As for deep learning projects, I prefer `pytorch` framework.

You should choose at least 5 repositories as the reference codebases.

I should use the determined repositories as reference codebases to implement the innovative ideas, so your decision should be as accurate as possible, and the number of repositories should be as less as possible. 

During the decision process, you can use the following tools:
1. You can use `execute_command` to git clone the repository to the working directory `/{working_dir}`. Choose 5-8 repositories you really need. And you should reserve the names of the repositories.

2. You can use `gen_code_tree_structure` to generate the tree structure of the code in the repository.

3. You can use `read_file` to read the content of the file in the repository. Note that read `README.md` file can help you know the purpose and function of the code in the repository, and read other files can help you know the details of the implementation.

4. You can use `terminal_page_down`, `terminal_page_up` and `terminal_page_to` to scroll the terminal output when it is too long. You can use `terminal_page_to` to move the viewport to the specific page of terminal where the meaningful content is, for example, when the terminal output contains a progress bar or output of generating directory structure when there are many datasets in the directory, you can use `terminal_page_to` to move the viewport to the end of terminal where the meaningful content is.

4. Finally, you should use the function `case_resolved` to output the determined reference codebases.
      """

    # 레지스트리에서 이름으로 도구를 조회하고,
    # 툴 시그니처에 'env'가 있을 때만 with_env_docker로 자동 주입됩니다.
    tool_names = [
        "gen_code_tree_structure",
        "read_file",
        "execute_command",
        "terminal_page_down",
        "terminal_page_up",
        "terminal_page_to",
    ]
    tools = get_tools(tool_names, env=code_env, env_wrapper=with_env_docker)  # 안전 조회/주입
    # 로컬 함수(case_resolved)는 레지스트리 외부 정의이므로 직접 추가
    tools.append(case_resolved)

    return Agent(
        name="Prepare Agent",
        model=model,
        instructions=instructions,
        functions=tools,
        tool_choice="required",
        parallel_tool_calls=False,
    )
