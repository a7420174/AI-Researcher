import datetime
import json
import asyncio
import argparse
import os
from typing import List, Dict, Any, Union, Optional

from tqdm import tqdm
from pydantic import BaseModel, Field

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import global_state

from research_agent.inno.workflow.flowcache import FlowModule, ToolModule, AgentModule
from research_agent.constant import DOCKER_WORKPLACE_NAME, COMPLETION_MODEL, CHEEP_MODEL
from research_agent.inno.util import single_select_menu, extract_json_from_output
from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig
from research_agent.inno.environment.local_env import LocalEnv, LocalConfig
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.logger import MetaChainLogger
from research_agent.inno.environment.utils import setup_dataset

from research_agent.inno.registry import get_agent_factory, get_tool

# Registry bootstrap
from app_bootstrap import bootstrap_registry

bootstrap_registry()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container_name", type=str, default="paper_eval")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--workplace_name", type=str, default="workplace")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--max_iter_times", type=int, default=0)
    parser.add_argument(
        "--use_docker",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use Docker for code execution (default: true)",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default=None,
        help="Path to conda installation (for local mode)",
    )
    parser.add_argument(
        "--use_conda",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Use conda for Python execution instead of uv (default: false)",
    )
    parser.add_argument(
        "--uv_path",
        type=str,
        default=None,
        help="Path to uv (default: uv)",
    )
    parser.add_argument(
        "--venv_path",
        type=str,
        default=None,
        help="Path to uv virtual environment (for uv mode)",
    )
    args = parser.parse_args()
    args.use_docker = args.use_docker.lower() == "true"
    args.use_conda = args.use_conda.lower() == "true"
    return args


class EvalMetadata(BaseModel):
    source_papers: str = Field(description="the reference papers string")
    task_instructions: str = Field(description="the task instructions")
    date: str = Field(description="the date", pattern="^\d{4}-\d{2}-\d{2}$")
    date_limit: str = Field(description="the date limit", pattern="^\d{4}-\d{2}-\d{2}$")


def load_instance_from_query(source_papers: str, task_instructions: str) -> Dict:
    """Query에서 직접 source_papers와 task_instructions를 받아 instance 생성"""
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    return EvalMetadata(
        source_papers=source_papers,
        task_instructions=task_instructions,
        date=date,
        date_limit=date,
    ).model_dump()


def github_search(metadata: Dict) -> str:
    search_github_repos = get_tool("search_github_repos")
    github_result = ""
    source_papers_str = metadata.get("source_papers", "")
    source_papers_list = [
        line.strip() for line in source_papers_str.strip().split("\n") if line.strip()
    ]
    for ref in tqdm(source_papers_list):
        github_result += search_github_repos(metadata, ref, 10)
        github_result += "*" * 30 + "\n"
    return github_result


# -------------------------------------------
# Flow 본체
# -------------------------------------------
class InnoFlow(FlowModule):
    def __init__(
        self,
        cache_path: str,
        log_path: Union[str, None, MetaChainLogger] = None,
        model: str = "gpt-4o-2024-08-06",
        code_env: Optional[DockerEnv] = None,
        file_env: Optional[RequestsMarkdownBrowser] = None,
    ):
        super().__init__(cache_path, log_path, model)

        self.code_env = code_env

        # ToolModule 바인딩
        self.git_search = ToolModule(github_search, cache_path)

        # ↓↓↓ registry에서 에이전트 팩토리 로드 후 인스턴스 생성
        # 첨부 ml_agent.py: get_ml_agent 등록 이름 확인됨 [1](https://amcsciences-my.sharepoint.com/personal/a7420174_amcsciences_com/Documents/Microsoft%20Copilot%20Chat%20%ED%8C%8C%EC%9D%BC/ml_agent.py)
        get_prepare_agent = get_agent_factory("get_prepare_agent")
        get_coding_plan_agent = get_agent_factory("get_coding_plan_agent")
        get_ml_agent = get_agent_factory("get_ml_agent")
        get_judge_agent = get_agent_factory("get_judge_agent")
        get_idea_agent = get_agent_factory("get_idea_agent")
        get_code_survey_agent = get_agent_factory("get_code_survey_agent")
        get_exp_analyser_agent = get_agent_factory("get_exp_analyser_agent")

        # registry 등록된 paper 다운로드 tool
        download_arxiv_source_by_title = get_tool("download_arxiv_source_by_title")

        # AgentModule 바인딩 (기존과 동일한 인터페이스)
        self.prepare_agent = AgentModule(
            get_prepare_agent(model=CHEEP_MODEL, code_env=code_env),
            self.client,
            cache_path,
        )
        self.download_papaer = ToolModule(download_arxiv_source_by_title, cache_path)
        self.coding_plan_agent = AgentModule(
            get_coding_plan_agent(model=CHEEP_MODEL, code_env=code_env),
            self.client,
            cache_path,
        )
        self.ml_agent = AgentModule(
            get_ml_agent(model=COMPLETION_MODEL, code_env=code_env),
            self.client,
            cache_path,
        )
        self.judge_agent = AgentModule(
            get_judge_agent(model=CHEEP_MODEL, code_env=code_env, file_env=file_env),
            self.client,
            cache_path,
        )
        self.idea_agent = AgentModule(
            get_idea_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env),
            self.client,
            cache_path,
        )
        # self.survey_agent    = AgentModule(get_survey_agent(model=CHEEP_MODEL, file_env=file_env, code_env=code_env), self.client, cache_path)
        self.code_survey_agent = AgentModule(
            get_code_survey_agent(
                model=CHEEP_MODEL, file_env=file_env, code_env=code_env
            ),
            self.client,
            cache_path,
        )
        self.exp_analyser = AgentModule(
            get_exp_analyser_agent(
                model=CHEEP_MODEL, file_env=file_env, code_env=code_env
            ),
            self.client,
            cache_path,
        )

    async def forward(
        self,
        local_root: Optional[str] = None,
        workplace_name: Optional[str] = None,
        max_iter_times: int = 0,
        references: Optional[str] = None,
        source_papers: Optional[str] = None,
        task_instructions: Optional[str] = None,
        *args,
        **kwargs,
    ):
        # Query 기반: source_papers 사용
        metadata = load_instance_from_query(source_papers, task_instructions or "")

        context_variables = {
            "working_dir": workplace_name,
            "date_limit": metadata["date_limit"],
        }

        # references가 비어있을 때 tool 실행 건너뛰기
        if not references:
            print("경고: references가 비어있습니다. tool 실행을 건너뜁니다.")
            prepare_res = "No papers provided."
            download_res = "No papers to download."
        else:
            github_result = self.git_search({"metadata": metadata})

            query = f"""\
You are given a list of papers, searching results of the papers on GitHub. 
List of papers:
{references}

Searching results of the papers on GitHub:
{github_result}

Your task is to choose at least 5 repositories as the reference codebases. Note that this time there is no innovative ideas, you should choose the most valuable repositories as the reference codebases.
"""
            messages = [{"role": "user", "content": query}]
            prepare_messages, context_variables = await self.prepare_agent(
                messages, context_variables
            )
            prepare_res = prepare_messages[-1]["content"]
            prepare_dict = extract_json_from_output(prepare_res)
            paper_list = prepare_dict.get("reference_papers", [])
            if not paper_list:
                print(
                    f"경고: reference_papers를 찾을 수 없음. prepare_dict: {prepare_dict}"
                )
                paper_list = []
            download_res = self.download_papaer(
                {
                    "paper_list": paper_list,
                    "local_root": local_root,
                    "workplace_name": workplace_name,
                }
            )

        idea_query = f"""\
I have a task related to machine learning:
{task_instructions}
And a list of papers for your reference:
{references}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}
And I have also downloaded the corresponding paper in the Tex format, with the following information:
{download_res}

Your task is to thoroughly review research papers and generate innovative ideas for the given task.

Note that the math formula should be as complete as possible.
"""
        messages = [{"role": "user", "content": idea_query}]
        context_variables["notes"] = []
        survey_messages, context_variables = await self.idea_agent(
            messages, context_variables
        )
        survey_res = survey_messages[-1]["content"]
        ideas = [survey_res]
        IDEA_NUM = 5
        for i in range(IDEA_NUM - 1):
            messages.extend(survey_messages)
            messages.append(
                {
                    "role": "user",
                    "content": "please survey again and give me another idea",
                }
            )
            survey_messages, context_variables = await self.idea_agent(
                messages, context_variables, iter_times=i + 1
            )
            survey_res = survey_messages[-1]["content"]
            ideas.append(survey_res)
        # messages.extend(survey_messages)
        messages = [
            {
                "role": "user",
                "content": """\
You have generated {} innovative ideas for the given task:
{}

Your task is to analyze multiple existing ideas, select the most novel one, enhance the idea if any key information is missing, finally give me the most novel idea with refined math formula and code implementation. Directly output the selected refined idea report.
""".format(IDEA_NUM, "\n===================\n===================".join(ideas)),
            }
        ]
        survey_messages, context_variables = await self.idea_agent(
            messages, context_variables, iter_times="select"
        )
        survey_res = survey_messages[-1]["content"]
        # print(survey_res)

        code_survey_query = f"""\
I have an innovative idea related to machine learning:
{survey_res}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, in the directory `/workplace`, use the `list_files` tool to navigate the directory.
And I have also downloaded the corresponding paper in the Tex format, with the following information:
{download_res}

Your task is to carefully understand the innovative idea, and thoroughly review codebases and generate a comprehensive implementation report for the innovative idea. You can NOT stop to review the codebases until you have get all academic concepts in the innovative idea.

Note that the code implementation should be as complete as possible.
"""
        messages = [{"role": "user", "content": code_survey_query}]
        code_survey_messages, context_variables = await self.code_survey_agent(
            messages, context_variables
        )
        code_survey_res = code_survey_messages[-1]["content"]
        # print(code_survey_res)

        context_variables["model_survey"] = code_survey_res

        plan_query = f"""\
I have an innovative ideas related to machine learning:
{survey_res}
And a list of papers for your reference:
{references}

I have carefully gone through these papers' github repositories and found download some of them in my local machine, with the following information:
{prepare_res}

I have also understood the innovative idea, comprehensively reviewed the codebases, and generated a comprehensive implementation report:
{code_survey_res}

Your task is to carefully review the existing resources and understand the task, and give me a detailed plan for the implementation.
"""
        messages = [{"role": "user", "content": plan_query}]
        plan_messages, context_variables = await self.coding_plan_agent(
            messages, context_variables
        )
        plan_res = plan_messages[-1]["content"]

        project_dir = (
            self.code_env.workplace + "/project"
            if self.code_env
            else f"/{workplace_name}/project"
        )
        workplace_dir = (
            self.code_env.workplace if self.code_env else f"/{workplace_name}"
        )
        workplace_dir = (
            self.code_env.docker_workplace if self.code_env else f"/{workplace_name}"
        )
        ml_dev_query = f"""\
INPUT:
You are given an innovative idea:
{survey_res}. 
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And I have conducted the comprehensive survey on the innovative idea and the papers, and give you the model survey notes:
{survey_res}
You should carefully go through the math formula and the code implementation, and implement the innovative idea according to the plan and existing resources.

Your task is to implement the innovative idea after carefully reviewing the math formula and the code implementation in the paper notes and existing resources in the directory `{workplace_dir}`. You should select ONE most appropriate and lightweight dataset from the given datasets, and implement the idea by creating new model, and EXACTLY run TWO epochs of training and testing on the ACTUAL dataset on the GPU device. Note that EVERY atomic academic concept in model survey notes should be implemented in the project.

PROJECT STRUCTURE REQUIREMENTS:
1. Directory Organization
- Data: `{project_dir}/data/`
     * Use the dataset selected by the `Plan Agent`
     * NO toy or random datasets
- Model Components: `{project_dir}/model/`
    * All model architecture files
    * All model components as specified in survey notes
    * Dataset processing scripts and utilities

- Training: `{project_dir}/training/`
    * Training loop implementation
    * Loss functions
    * Optimization logic

- Testing: `{project_dir}/testing/`
    * Evaluation metrics
    * Testing procedures

- Data processing: `{project_dir}/data_processing/`
    * Implement the data processing pipeline

- Main Script: `{project_dir}/run_training_testing.py`
    * Complete training and testing pipeline
    * Configuration management
    * Results logging

2. Complete Implementation Requirements
   - MUST implement EVERY component from model survey notes
   - NO placeholder code (no `pass`, `...`, `raise NotImplementedError`)
   - MUST include complete logic and mathematical operations
   - Each component MUST be fully functional and tested

3. Dataset and Training Requirements
   - Select and download ONE actual dataset from references
   - Implement full data processing pipeline
   - Train for exactly 2 epochs
   - Test model performance after training
   - Log all metrics and results

4. Integration Requirements
   - All components must work together seamlessly
   - Clear dependencies between modules
   - Consistent coding style and documentation
   - Proper error handling and GPU support

EXECUTION WORKFLOW:
1. Dataset Setup
   - Choose appropriate dataset from references (You MUST use the actual dataset, not the toy or random datasets) [IMPORTANT!!!]
   - Download to data directory `{project_dir}/data`
   - Implement processing pipeline in `{project_dir}/data_processing/`
   - Verify data loading

2. Model Implementation
   - Study model survey notes thoroughly
   - Implement each component completely
   - Document mathematical operations
   - Add comprehensive docstrings

3. Training Implementation
   - Complete training loop
   - Loss function implementation
   - Optimization setup
   - Progress monitoring

4. Testing Setup
   - Implement evaluation metrics
   - Create testing procedures
   - Set up results logging
   - Error handling

5. Integration
   - Create run_training_testing.py
   - Configure for 2 epoch training
   - Add GPU support and OOM handling
   - Implement full pipeline execution

VERIFICATION CHECKLIST:
1. Project Structure
   - All directories exist and are properly organized
   - Each component is in correct location
   - Clear separation of concerns

2. Implementation Completeness
   - Every function is fully implemented
   - No placeholder code exists
   - All mathematical operations are coded
   - Documentation is complete

3. Functionality
   - Dataset downloads and loads correctly
   - Training runs for 2 epochs
   - Testing produces valid metrics
   - GPU support is implemented

Remember: 
- MUST use actual dataset (no toy data, download according to the reference codebases) [IMPORTANT!!!]
- Implementation MUST strictly follow model survey notes
- ALL components MUST be fully implemented
- Project MUST run end-to-end without placeholders
- MUST complete 2 epochs of training and testing
"""
        messages = [{"role": "user", "content": ml_dev_query}]
        ml_dev_messages, context_variables = await self.ml_agent(
            messages, context_variables
        )
        ml_dev_res = ml_dev_messages[-1]["content"]

        query = f"""\
INPUT:
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
The implementation of the project:
{ml_dev_res}
Your task is to evaluate the implementation, and give a suggestion about the implementation. Note that you should carefully check whether the implementation meets the idea, especially the atomic academic concepts in the model survey notes one by one! If not, give comprehensive suggestions about the implementation.

[IMPORTANT] You should fully utilize the existing resources in the reference codebases as much as possible, including using the existing datasets, model components, and training process, but you should also implement the idea by creating new model components!

[IMPORTANT] You should recognize every key point in the innovative idea, and carefully check whether the implementation meets the idea one by one!

[IMPORTANT] Some tips about the evaluation:
1. The implementation should carefully follow the plan. Please check every component in the plan step by step.
2. The implementation should have the test process. All in all, you should train ONE dataset with TWO epochs, and finally test the model on the test dataset within one script. The test metrics should follow the plan.
3. The model should be train on GPU device. If you meet Out of Memory problem, you should try another specific GPU device.
"""
        input_messages = [{"role": "user", "content": query}]
        judge_messages, context_variables = await self.judge_agent(
            input_messages, context_variables
        )
        judge_res = judge_messages[-1]["content"]

        MAX_ITER_TIMES = max_iter_times
        for i in range(MAX_ITER_TIMES):
            query = f"""\
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
And your last implementation of the project:
{ml_dev_res}
The suggestion about your last implementation:
{judge_res}
Your task is to modify the project according to the suggestion. Note that you should MODIFY rather than create a new project! Take full advantage of the existing resources! Still use the SAME DATASET!

[IMPORTANT] You should modify the project in the directory `{project_dir}`, rather than create a new project!

[IMPORTANT] If you meet dataset missing problem, you should download the dataset from the reference codebases, and put the dataset in the directory `{project_dir}/data`. 

[IMPORTANT] You CANNOT stop util you 2 epochs of training and testing on your model with the ACTUAL dataset.

[IMPORTANT] You encounter ImportError while using `run_python()`, you should check whether every `__init__.py` file is correctly implemented in the directories in the `{project_dir}`!

[IMPORTANT] Carefully check whether model and its components are correctly implemented according to the model survey notes!

Remember: 
- Implementation MUST strictly follow model survey notes
- ALL components MUST be fully implemented
- Project MUST run end-to-end without placeholders
- MUST use actual dataset (no toy data)
- MUST complete 2 epochs of training and testing
"""
            judge_messages.append({"role": "user", "content": query})
            judge_messages, context_variables = await self.ml_agent(
                judge_messages, context_variables, iter_times=i + 1
            )
            ml_dev_res = judge_messages[-1]["content"]

            query = f"""\
You are given an innovative idea:
{survey_res}
and the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
and the detailed coding plan:
{plan_res}
The implementation of the project:
{ml_dev_res}
Please evaluate the implementation, and give a suggestion about the implementation.
"""
            judge_messages.append({"role": "user", "content": query})
            judge_messages, context_variables = await self.judge_agent(
                judge_messages, context_variables, iter_times=i + 1
            )
            judge_res = judge_messages[-1]["content"]
            if '"fully_correct": true' in judge_messages[-1]["content"]:
                break

        ml_submit_query = f"""\
You are given an innovative idea:
{survey_res}
And your last implementation of the project:
{ml_dev_res}
The suggestion about your last implementation:
{judge_res}
You have run out the maximum iteration times to implement the idea by running the script `run_training_testing.py` with TWO epochs of training and testing on ONE ACTUAL dataset.
Your task is to submit the code to the environment by running the script `run_training_testing.py` with APPROPRIATE epochs of training and testing on THIS ACTUAL dataset in order to get some stastical results. You must MODIFY the epochs in the script `run_training_testing.py` rather than use the 2 epochs.

[IMPORTANT] In this stage, you are NOT allowed to modify the existing code in the script `run_training_testing.py` except for the epochs!

Note that if your last implementation is not runable, you should finalize the submission with `case_not_resolved` function. But you can temporarily ignore the judgement of the `Judge Agent` which contains the suggestions about the implementation.
After you get the result, you should return the result with your analysis and suggestions about the implementation with `case_resolved` function.
"""
        judge_messages.append({"role": "user", "content": ml_submit_query})
        judge_messages, context_variables = await self.ml_agent(
            judge_messages, context_variables, iter_times="submit"
        )
        submit_res = judge_messages[-1]["content"]

        EXP_ITER_TIMES = 2
        for i in range(EXP_ITER_TIMES):
            exp_planner_query = f"""\
You are given an innovative idea:
{survey_res}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{submit_res}
Your task is to: 
1. Analyze the experimental results and give a detailed analysis report about the results.
2. Analyze the reference codebases and papers, and give a further plan to let `Machine Learning Agent` to do more experiments based on the innovative idea. The further experiments could include but not limited to:
    - Modify the implementation to better fit the idea.
    - Add more experiments to prove the effectiveness and superiority of the idea. 
    - Visualize the experimental results and give a detailed analysis report about the results.
    - ANY other experiments that exsiting concurrent reference papers and codebases have done.
DO NOT use the `case_resolved` function before you have carefully and comprehensively analyzed the experimental results and the reference codebases and papers.
"""
            judge_messages.append({"role": "user", "content": exp_planner_query})
            judge_messages, context_variables = await self.exp_analyser(
                judge_messages, context_variables, iter_times=f"refine_{i + 1}"
            )

            analysis_report = context_variables["experiment_report"][-1][
                "analysis_report"
            ]
            further_plan = context_variables["experiment_report"][-1]["further_plan"]
            # print(analysis_report)
            refine_query = f"""\
You are given an innovative idea:
{survey_res}
And the reference codebases chosen by the `Prepare Agent`:
{prepare_res}
And the detailed coding plan:
{plan_res}
You have conducted the experiments and get the experimental results:
{submit_res}
And a detailed analysis report about the results are given by the `Experiment Planner Agent`:
{analysis_report}
Your task is to refine the experimental results according to the analysis report by modifying existing code in the directory `{project_dir}`. You should NOT stop util every experiment is done with ACTUAL results. If you encounter Out of Memory problem, you should try another specific GPU device. If you encounter ANY other problems, you should try your best to solve the problem by yourself.

Note that you should fully utilize the existing code in the directory `{project_dir}` as much as possible. If you want to add more experiments, you should add the python script in the directory `{project_dir}/`, like `run_training_testing.py`. Select and output the important results during the experiments into the log files, do NOT output them all in the terminal.
"""
            judge_messages.append({"role": "user", "content": refine_query})
            judge_messages, context_variables = await self.ml_agent(
                judge_messages, context_variables, iter_times=f"refine_{i + 1}"
            )
            refine_res = judge_messages[-1]["content"]

        print(refine_res)


def main(args, references, task_instructions=None):
    """
    Args:
        args: arguments containing container_name, workplace_name, port, model, category, etc.
        references: reference papers string
        task_instructions: task instructions string (for query-based mode)

    Query-based mode: Pass references and task_instructions directly as strings

    Returns:
        dict: Project information including agent_dir and model_dir paths
    """
    use_docker = getattr(args, "use_docker", True)

    # Query 기반 모드
    instance_id = "query_based"

    # Get project root (parent of research_agent)
    if hasattr(args, "project_root") and args.project_root:
        project_root = args.project_root
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_root = os.path.join(
        project_root,
        "workplace_paper",
        f"task_{instance_id}"
        + "_"
        + COMPLETION_MODEL.replace("/", "__").replace(":", "_"),
    )
    os.makedirs(local_root, exist_ok=True)

    code_env = None
    file_env = None

    if use_docker:
        container_name = (
            args.container_name
            + "_"
            + instance_id
            + "_"
            + COMPLETION_MODEL.replace("/", "__").replace(":", "_")
        )

        env_config = DockerConfig(
            container_name=container_name,
            workplace_name=args.workplace_name,
            communication_port=args.port,
            local_root=local_root,
        )
        code_env = DockerEnv(env_config)
        code_env.init_container()
        global_state.CODE_ENV = code_env

        file_env = RequestsMarkdownBrowser(
            viewport_size=1024 * 4,
            local_root=env_config.local_root,
            workplace_name=env_config.workplace_name,
            downloads_folder=os.path.join(
                env_config.local_root, env_config.workplace_name, "downloads"
            ),
        )
    else:
        print("[INFO] Running in LOCAL mode (no Docker)")
        local_config = LocalConfig(
            workplace_name=args.workplace_name,
            local_root=local_root,
            conda_path=args.conda_path,
            use_uv=not args.use_conda,
            uv_path=args.uv_path,
            venv_path=args.venv_path,
        )
        code_env = LocalEnv(local_config)
        code_env.init_local()
        global_state.CODE_ENV = code_env

        file_env = RequestsMarkdownBrowser(
            viewport_size=1024 * 4,
            local_root=local_root,
            workplace_name=args.workplace_name,
            downloads_folder=os.path.join(local_root, args.workplace_name, "downloads"),
        )

    flow = InnoFlow(
        cache_path=os.path.join(
            project_root,
            "workplace_paper",
            "cache_"
            + instance_id
            + "_"
            + COMPLETION_MODEL.replace("/", "__").replace(":", "_"),
        ),
        log_path="log_" + instance_id,
        code_env=code_env,
        file_env=file_env,
        model=args.model,
    )

    # Query 기반: references와 task_instructions 문자열 직접 전달
    flow_kwargs = {
        "local_root": local_root,
        "workplace_name": args.workplace_name,
        "max_iter_times": args.max_iter_times,
        "references": references,
        "source_papers": references,
        "task_instructions": task_instructions,
    }

    asyncio.run(flow(**flow_kwargs))

    # Return project information for writing module
    agent_dir = os.path.join(local_root, args.workplace_name)
    model_dir = os.path.join(local_root, args.workplace_name, "project")

    return {
        "instance_id": instance_id,
        "local_root": local_root,
        "agent_dir": agent_dir,
        "model_dir": model_dir,
        "workplace_name": args.workplace_name,
    }


if __name__ == "__main__":
    args = get_args()
    # 외부에서 references를 구성해 넘겨주세요. (원 코드와 동일)
    main(args, references="")
