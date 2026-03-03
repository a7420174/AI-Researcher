"""
Deep Research Module - Using Deep Survey Agent
This module uses the Deep Survey Agent for comprehensive research with verification.
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from research_agent.inno.registry import get_agent_factory
from research_agent.constant import COMPLETION_MODEL, CHEEP_MODEL
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.repl.repl import MetaChain

# Registry bootstrap
from app_bootstrap import bootstrap_registry

bootstrap_registry()


class DeepResearchFlow:
    def __init__(
        self,
        cache_path: str = None,
        log_path: str = None,
        model: str = "gpt-4o-2024-08-06",
    ):
        self.cache_path = cache_path
        self.model = model
        self.log_path = log_path
        self.file_env: Optional[RequestsMarkdownBrowser] = None

    def _get_file_env(self) -> RequestsMarkdownBrowser:
        """Get or create file environment."""
        if self.file_env is None:
            self.file_env = RequestsMarkdownBrowser(
                local_root="/tmp",
                workplace_name="research",
            )
        return self.file_env

    async def research(
        self,
        topic: str,
        max_search_results: int = 10,
        verify: bool = True,
    ) -> str:
        """
        Perform deep research on a given topic using Deep Survey Agent.

        Args:
            topic: The research topic/question
            max_search_results: Maximum number of search results to gather
            verify: Whether to run verification and fix iteration (default: True)

        Returns:
            Comprehensive research findings
        """
        try:
            get_deep_survey_agent = get_agent_factory("get_deep_survey_agent")
        except Exception as e:
            return f"Error: Deep Survey Agent not available: {str(e)}"

        file_env = self._get_file_env()
        agent = get_deep_survey_agent(model=self.model, file_env=file_env)

        research_prompt = f"""Please perform comprehensive research on the following topic:

Topic: {topic}

IMPORTANT SEARCH INSTRUCTIONS:
1. When searching for drug/trial/article databases, ALWAYS use the EXACT topic terms.
   - For example, if the topic is "IL1RAP ADC", search for "IL1RAP ADC" specifically,
     NOT just "ADC" or "IL1RAP" alone.
   - Use compound search terms like "IL1RAP ADC", "IL1RAP antibody-drug conjugate"
2. Search WITHOUT year limits to get the latest information (do NOT restrict to 2024)
3. After completing research, you MUST call transfer_to_judge_agent automatically,
   do NOT ask the user - just proceed with the transfer.

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
Then automatically call transfer_to_judge_agent with the final research summary.
"""

        try:
            client = MetaChain()
            response = client.run(
                agent=agent,
                messages=[{"role": "user", "content": research_prompt}],
                context_variables={},
            )

            if hasattr(response, "messages") and response.messages:
                last_message = response.messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                return str(last_message)
            return str(response)
        except Exception as e:
            return f"Error during research: {str(e)}"


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

    # Generate instance_id from model (same pattern as run_infer_plan)
    instance_id = f"deep_research"

    # Get project root (parent of research_agent)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    async def run_research():
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
            model=COMPLETION_MODEL,
        )
        result = await flow.research(topic=topic, verify=verify)
        return result

    research_result = asyncio.run(run_research())

    # Create local_root (same pattern as run_infer_plan)
    local_root = os.path.join(
        project_root,
        "workplace_paper",
        f"task_{instance_id}"
        + "_"
        + COMPLETION_MODEL.replace("/", "__").replace(":", "_"),
    )
    os.makedirs(local_root, exist_ok=True)

    # Return research result and project info
    return {
        "result": research_result,
        "instance_id": instance_id,
        "local_root": local_root,
        "agent_dir": None,
        "model_dir": None,
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
