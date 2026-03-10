"""
Deep Research Module - Using Deep Survey Agent
This module uses the Deep Survey Agent for comprehensive research with verification.
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Union, Optional
from dotenv import load_dotenv

from research_agent.inno.workflow.flowcache import FlowModule, AgentModule, ToolModule
from research_agent.inno.registry import get_agent_factory, get_tool
from research_agent.constant import COMPLETION_MODEL, CHEEP_MODEL
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.logger import MetaChainLogger

# Registry bootstrap
from app_bootstrap import bootstrap_registry

bootstrap_registry()


class DeepResearchFlow(FlowModule):
    def __init__(
        self,
        cache_path: str,
        log_path: Union[str, None, MetaChainLogger] = None,
        model: str = "gpt-4o-2024-08-06",
        file_env: Optional[RequestsMarkdownBrowser] = None,
    ):
        super().__init__(cache_path, log_path, model)
        self.file_env = file_env

        get_deep_survey_agent = get_agent_factory("get_deep_survey_agent")

        self.survey_agent = AgentModule(
            get_deep_survey_agent(model=model, file_env=file_env),
            self.client,
            cache_path,
        )

    async def forward(
        self,
        topic: Optional[str] = None,
        max_iter_times: int = 1,
        *args,
        **kwargs,
    ):
        context_variables = {}

        MAX_ITER_TIMES = max_iter_times
        survey_res = ""
        suggestion_text = ""

        for i in range(MAX_ITER_TIMES):
            # 이전 iteration의 suggestion을 현재 프롬프트에 포함
            suggestion_prefix = (
                f"\n\nPrevious iteration suggestions:\n{suggestion_text}\n\nPlease address these suggestions in your research.\n"
                if suggestion_text
                else ""
            )
            research_prompt = f"""Please perform comprehensive research on the following topic:

Topic: {topic}{suggestion_prefix}"""

            messages = [{"role": "user", "content": research_prompt}]

            survey_messages, context_variables = await self.survey_agent(
                messages, context_variables
            )

            if not survey_messages:
                continue

            research_content = ""
            for msg in reversed(survey_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if content and "<final_answer>" not in content:
                        research_content = content
                        break

            if not research_content:
                research_content = survey_messages[-1].get("content", "")

            survey_res = research_content

            if not survey_res or "<final_answer>" in survey_res:
                continue

            # suggestion 추출 - fully_correct가 false면 suggestion을 다음 iteration에 전달
            suggestion_dict = context_variables.get("suggestion_dict", {})
            fully_correct = suggestion_dict.get("fully_correct", False)

            if fully_correct and survey_res:
                break

            suggestion_text = suggestion_dict.get("suggestion", "")

        return {
            "survey_result": survey_res,
            "citations": context_variables.get("citations", ""),
        }


def main(topic: str, max_iter_times: int = 1):
    """
    Main entry point for Deep Research using Agent.

    Args:
        topic: The research topic/question
        max_iter_times: Maximum of iteration times

    Returns:
        dict: Contains 'result' (research findings) and project paths
    """
    if not topic:
        return {"error": "Error: Topic is required for deep research", "result": None}

    instance_id = f"deep_research"

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_root = os.path.join(
        project_root,
        "workplace_paper",
        f"task_{instance_id}"
        + "_"
        + COMPLETION_MODEL.replace("/", "__").replace(":", "_"),
    )
    os.makedirs(local_root, exist_ok=True)

    file_env = RequestsMarkdownBrowser(
        local_root=local_root,
        workplace_name="workplace",
    )

    flow = DeepResearchFlow(
        cache_path=os.path.join(
            project_root,
            "workplace_paper",
            "cache_"
            + instance_id
            + "_"
            + COMPLETION_MODEL.replace("/", "__").replace(":", "_"),
        ),
        log_path="log_" + instance_id,
        file_env=file_env,
        model=COMPLETION_MODEL,
    )

    result = asyncio.run(
        flow(
            topic=topic,
            max_iter_times=max_iter_times,
        )
    )

    workplace_name = "workplace"
    agent_dir = os.path.join(local_root, workplace_name)
    model_dir = os.path.join(local_root, workplace_name, "project")

    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(agent_dir, "research_result.md"), "w") as f:
        f.write(f"# Research Topic: {topic}\n\n")
        f.write(result.get("survey_result", ""))

    # Save related papers search results to JSON for paper_agent
    related_papers_data = {
        "topic": topic,
        "search_results": result.get("related_papers_search", ""),
    }
    with open(
        os.path.join(agent_dir, "related_papers.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(related_papers_data, f, ensure_ascii=False, indent=2)

    return {
        "result": result.get("survey_result", ""),
        "instance_id": instance_id,
        "local_root": local_root,
        "agent_dir": agent_dir,
        "model_dir": model_dir,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research Module using Agent")
    parser.add_argument(
        "--topic", type=str, required=True, help="Research topic/question"
    )
    parser.add_argument("--max-iter-times", type=int, default=5)

    args = parser.parse_args()

    result = main(topic=args.topic, max_iter_times=args.max_iter_times)
    print(result)
