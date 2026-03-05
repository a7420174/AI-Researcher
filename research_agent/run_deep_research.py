"""
Deep Research Module - Using Deep Survey Agent
This module uses the Deep Survey Agent for comprehensive research with verification.
"""

import os
import asyncio
from typing import Dict, Any, List, Union, Optional
from dotenv import load_dotenv

from research_agent.inno.workflow.flowcache import FlowModule, AgentModule
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
        cache_path: str = None,
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
        topic: str = None,
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
            suggestion_prefix = f"\n\nPrevious iteration suggestions:\n{suggestion_text}\n\nPlease address these suggestions in your research.\n" if suggestion_text else ""
            research_prompt = f"""Please perform comprehensive research on the following topic:

Topic: {topic}{suggestion_prefix}

IMPORTANT SEARCH INSTRUCTIONS:
1. When searching for drug/trial/article databases, ALWAYS use the EXACT topic terms.
   - For example, if the topic is "IL1RAP ADC", search for "IL1RAP ADC" specifically,
     NOT just "ADC" or "IL1RAP" alone.
   - Use compound search terms like "IL1RAP ADC", "IL1RAP antibody-drug conjugate"
2. Search WITHOUT year limits to get the latest information (do NOT restrict to 2024)

Research Requirements:
1. Search and analyze information from BioMCP (biomedical databases) - use EXACT topic terms
2. Search academic papers from OpenAlex - use EXACT topic terms, no year limits
3. Search the web for additional information - use EXACT topic terms
4. Synthesize findings into a comprehensive summary

Please provide:
- Executive Summary
- Key findings from biomedical databases
- Key findings from academic literature
- Key findings from web searches
- Clinical trial information
- Sources and references

After reviewing, use the `case_resolved` function to provide your final verdict:
- Set `fully_correct` to True if the research is satisfactory
- If not fully correct, provide suggestions for improvement
- From the suggestions, verify and fix any issues found.
Then provide the final research summary."""

            messages = [{"role": "user", "content": research_prompt}]

            survey_messages, context_variables = await self.survey_agent(
                messages, context_variables
            )
            survey_res = survey_messages[-1]["content"]
            context_variables["model_survey"] = survey_res
            
            # suggestion 추출 - fully_correct가 false면 suggestion을 다음 iteration에 전달
            suggestion_dict = context_variables.get("suggestion_dict", {})
            if suggestion_dict.get("fully_correct", False):
                break
            
            # suggestion이 있으면 추출하여 다음 iteration에 전달
            suggestion_text = suggestion_dict.get("suggestion", "")

        return {
            "survey_result": survey_res,
        }


def main(
    topic: str, max_iter_times: int = 1
):
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

    result = asyncio.run(flow(
        topic=topic,
        max_iter_times=max_iter_times,
    ))

    workplace_name = "workplace"
    agent_dir = os.path.join(local_root, workplace_name)
    model_dir = os.path.join(local_root, workplace_name, "project")

    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(agent_dir, "research_result.md"), "w") as f:
        f.write(f"# Research Topic: {topic}\n\n")
        f.write(result.get("survey_result", ""))

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
