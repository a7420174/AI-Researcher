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
        get_judge_agent = get_agent_factory("get_judge_agent")

        self.survey_agent = AgentModule(
            get_deep_survey_agent(model=model, file_env=file_env),
            self.client,
            cache_path,
        )
        self.judge_agent = AgentModule(
            get_judge_agent(model=model, file_env=file_env, code_env=None),
            self.client,
            cache_path,
        )

    async def forward(
        self,
        topic: str = None,
        *args,
        **kwargs,
    ):
        context_variables = {}

        research_prompt = f"""Please perform comprehensive research on the following topic:

Topic: {topic}

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

After providing the initial summary, verify and fix any issues found (up to 3 iterations).
Then provide the final research summary."""

        messages = [{"role": "user", "content": research_prompt}]

        max_retries = 3
        retry_count = 0
        last_res = None
        survey_res = None

        while retry_count < max_retries:
            survey_messages, context_variables = await self.survey_agent(
                messages, context_variables
            )
            current_res = survey_messages[-1]["content"]

            if current_res == last_res:
                retry_count += 1
                if retry_count >= max_retries:
                    survey_res = current_res
                    break
            else:
                retry_count = 0
                survey_res = current_res
                last_res = current_res
                if (
                    "final verified research summary" in current_res.lower()
                    or "research completed" in current_res.lower()
                ):
                    break

            messages = [
                {
                    "role": "user",
                    "content": "Please continue with the research and provide the final summary.",
                }
            ]

        if survey_res is None:
            survey_res = (
                last_res if last_res else "Research completed but no summary available."
            )

        judge_prompt = f"""Please review the following research summary for accuracy, completeness, and quality.

Research Summary to Review:
{survey_res}

After reviewing, use the `case_resolved` function to provide your final verdict:
- Set `fully_correct` to True if the research is satisfactory
- Set `review_type` to "response_review"
- If not fully correct, provide suggestions for improvement"""

        input_messages = [{"role": "user", "content": judge_prompt}]
        judge_messages, context_variables = await self.judge_agent(
            input_messages, context_variables
        )

        if '"fully_correct": true' in judge_messages[-1]["content"]:
            judge_res = judge_messages[-1]["content"]
        else:
            judge_res = judge_messages[-1]["content"]

        return {
            "survey_result": survey_res,
            "judge_result": judge_res,
        }


def main(
    topic: str, reference: str = None, use_docker: bool = None, verify: bool = True
):
    """
    Main entry point for Deep Research using Agent.

    Args:
        topic: The research topic/question
        reference: Optional reference papers (not used in Deep Research mode)
        use_docker: Whether to use Docker (not used - Deep Research is Docker-independent)
        verify: Whether to run verification and fix iteration (default: True)

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

    result = asyncio.run(flow(topic=topic))

    workplace_name = "workplace"
    agent_dir = os.path.join(local_root, workplace_name)
    model_dir = os.path.join(local_root, workplace_name, "project")

    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(agent_dir, "research_result.md"), "w") as f:
        f.write(f"# Research Topic: {topic}\n\n")
        f.write(result.get("survey_result", ""))
        f.write("\n\n## Judge Review\n\n")
        f.write(result.get("judge_result", ""))

    return {
        "result": result.get("survey_result", ""),
        "judge_result": result.get("judge_result", ""),
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
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use (default: CHEEP_MODEL)"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )

    args = parser.parse_args()

    result = main(topic=args.topic, verify=not args.no_verify)
    print(result)
