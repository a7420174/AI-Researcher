"""
Deep Research Module - Web-based research without ML implementation
This module performs comprehensive web research using search tools, BioMCP, and LLM.
"""

import os
import asyncio
import re
from typing import Dict, Any, Union, List, Optional
from dotenv import load_dotenv
import litellm

from research_agent.inno.registry import get_tool
from research_agent.constant import COMPLETION_MODEL, CHEEP_MODEL, DOCKER_WORKPLACE_NAME

try:
    from research_agent.inno.environment.browser_env import BrowserEnv
    from research_agent.inno.environment.docker_env import DockerEnv, DockerConfig

    BROWSER_ENV_AVAILABLE = True
except ImportError as e:
    BROWSER_ENV_AVAILABLE = False
    BrowserEnv = None
    DockerEnv = None
    DockerConfig = None


class DeepResearchFlow:
    def __init__(
        self,
        cache_path: str = None,
        log_path: Union[str, None] = None,
        model: str = "gpt-4o-2024-08-06",
        web_env=None,
    ):
        self.cache_path = cache_path
        self.model = model
        self.log_path = log_path
        self.web_env = web_env

    async def research(
        self,
        topic: str,
        max_search_results: int = 10,
    ) -> str:
        """
        Perform deep research on a given topic using web search, BioMCP, and LLM.

        Args:
            topic: The research topic/question
            max_search_results: Maximum number of search results to gather

        Returns:
            Comprehensive research findings
        """
        topic_analysis = await self._analyze_topic(topic)

        biomcp_results = await self._run_biomcp_searches(topic, topic_analysis)

        web_results = await self._run_web_searches(topic, topic_analysis)

        all_results = {
            "biomcp": biomcp_results,
            "web": web_results,
        }

        research_context = self._format_all_results(all_results)

        synthesis_query = f"""You are a research assistant. Based on the following research results, provide a comprehensive research summary on the topic:

Topic: {topic}

{"=" * 60}
BIOMEDICAL DATABASE RESULTS (BioMCP):
{"=" * 60}
{research_context.get("biomcp", "No BioMCP results")}

{"=" * 60}
WEB SEARCH RESULTS:
{"=" * 60}
{research_context.get("web", "No web results")}

Please provide:
1. Executive Summary
2. Key findings and information from biomedical databases
3. Key findings from web searches
4. Comparison and synthesis of all sources
5. Relevant details (companies, products, development stages, indications, etc.)
6. Sources and references
7. Any gaps or areas needing further research

Format your response in a well-structured manner with clear sections. Use tables where appropriate.
"""

        messages = [{"role": "user", "content": synthesis_query}]

        try:
            response = await litellm.acompletion(
                model=CHEEP_MODEL,
                messages=messages,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error during research synthesis: {str(e)}\n\nRaw results:\n{research_context}"

    async def _analyze_topic(self, topic: str) -> Dict[str, Any]:
        """Analyze the topic using LLM to extract biomedical entities."""

        analysis_prompt = f"""You are a biomedical research assistant. Analyze the following research topic and extract key entities for database searching.

Topic: {topic}

Extract the following information in JSON format:
1. genes: List of gene symbols or protein names mentioned (e.g., IL1RAP, KRAS, EGFR, PD-1)
2. drugs: List of drug names or drug classes (e.g., ADC, CAR-T, inhibitor, antibody)
3. diseases: List of diseases or conditions (e.g., lung cancer, leukemia, melanoma)
4. keywords: Important biomedical keywords that should be searched
5. is_biomedical: Boolean indicating if this is a biomedical topic (true if it mentions genes, proteins, drugs, diseases, treatments, etc.)

Return a JSON object with these fields. Be thorough - extract ALL relevant genes, drugs, and diseases from the topic.

Example output format:
{{
    "genes": ["IL1RAP", "KRAS"],
    "drugs": ["ADC", "inhibitor"],
    "diseases": ["lung cancer"],
    "keywords": ["antibody-drug conjugate", "targeted therapy"],
    "is_biomedical": true
}}
"""

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3,
            )
            content = response.choices[0].message.content

            import json
            import re

            json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", content)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())

                    analysis.setdefault("genes", [])
                    analysis.setdefault("drugs", [])
                    analysis.setdefault("diseases", [])
                    analysis.setdefault("keywords", [])
                    analysis.setdefault("is_biomedical", False)

                    return analysis
                except json.JSONDecodeError as je:
                    print(f"JSON parse error: {je}")
            else:
                print(f"No JSON found in response: {content[:200]}")
        except Exception as e:
            print(f"Error in LLM topic analysis: {e}")

        return {
            "genes": [],
            "drugs": [],
            "diseases": [],
            "keywords": [],
            "is_biomedical": True,
        }

    async def _run_biomcp_searches(self, topic: str, analysis: Dict) -> Dict[str, str]:
        """Run BioMCP database searches based on topic analysis."""
        results = {}

        if not analysis["is_biomedical"] and not analysis["genes"]:
            return results

        biomcp_tools = []

        if analysis["genes"]:
            gene = analysis["genes"][0]
            biomcp_tools.extend(
                [
                    ("biomcp_gene_search", {"gene": gene, "limit": 5}),
                    ("biomcp_article_search", {"gene": gene, "limit": 10}),
                    ("biomcp_trial_search", {"term": gene, "limit": 10}),
                ]
            )

        if analysis["diseases"]:
            disease = analysis["diseases"][0]
            if analysis["genes"]:
                gene = analysis["genes"][0]
                biomcp_tools.append(
                    (
                        "biomcp_article_search",
                        {"gene": gene, "disease": disease, "limit": 10},
                    )
                )
                biomcp_tools.append(
                    (
                        "biomcp_trial_search",
                        {"condition": disease, "intervention": gene, "limit": 10},
                    )
                )
            else:
                biomcp_tools.append(
                    ("biomcp_article_search", {"disease": disease, "limit": 10})
                )
                biomcp_tools.append(
                    ("biomcp_trial_search", {"condition": disease, "limit": 10})
                )

        if analysis.get("keywords"):
            for keyword in analysis["keywords"]:
                biomcp_tools.append(
                    ("biomcp_article_search", {"keyword": keyword, "limit": 5})
                )

        if not biomcp_tools and analysis["is_biomedical"]:
            main_keyword = (
                analysis["genes"][0] if analysis["genes"] else topic.split()[0]
            )
            biomcp_tools.append(
                ("biomcp_article_search", {"keyword": main_keyword, "limit": 5})
            )

        for tool_name, params in biomcp_tools:
            try:
                tool = get_tool(tool_name)
                result = tool(**params)
                results[
                    f"{tool_name}_{params.get('gene', params.get('keyword', params.get('disease', 'default')))}"
                ] = result
            except Exception as e:
                results[tool_name] = f"Error: {str(e)}"

        return results

    async def _run_web_searches(
        self, topic: str, analysis: Dict = None
    ) -> Dict[str, str]:
        """Run OpenAlex searches for the topic (academic papers)."""
        results = {}

        search_queries = await self._generate_search_queries(topic, analysis)

        try:
            from research_agent.inno.tools import openalex_tools as openalex_module

            search_tool = get_tool("openalex_search_papers")
        except Exception as e:
            for query in search_queries:
                results[query] = f"Error: OpenAlex tools not available: {e}"
            return results

        for query in search_queries:
            try:
                result = search_tool(
                    query=query,
                    max_results=20,
                )
                results[query] = result
            except Exception as e:
                results[query] = f"Error: {str(e)}"

        return results

    async def _generate_search_queries(
        self, topic: str, analysis: Dict = None
    ) -> List[str]:
        """Generate dynamic search queries based on the topic using LLM."""

        query_prompt = f"""Generate effective web search queries for the following research question.

Research Question: {topic}

Extracted Entities:
- Genes: {analysis.get("genes", []) if analysis else []}
- Drugs: {analysis.get("drugs", []) if analysis else []}
- Diseases: {analysis.get("diseases", []) if analysis else []}
- Keywords: {analysis.get("keywords", []) if analysis else []}

Generate 8-12 search queries that would help answer this research question. Include:
1. General overview queries
2. Specific queries about companies, products, development status
3. Clinical trial information
4. Recent research (don't limit to specific years - search all available)
5. Queries about the specific genes/drugs/diseases mentioned

Return ONLY a JSON array of strings (no explanation):
["query 1", "query 2", ...]
"""

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": query_prompt}],
                temperature=0.5,
            )
            content = response.choices[0].message.content

            import json
            import re

            json_match = re.search(r"\[[\s\S]*\]", content)
            if json_match:
                queries = json.loads(json_match.group())
                return queries[:12]
        except Exception as e:
            print(f"Error generating search queries: {e}")

        return [
            topic,
            f"{topic} overview",
            f"{topic} development status",
            f"{topic} clinical trial",
            f"{topic} company pipeline",
        ]

    def _generate_search_queries_sync(
        self, topic: str, analysis: Dict = None
    ) -> List[str]:
        """Synchronous fallback for search query generation."""
        return [
            topic,
            f"{topic} overview",
            f"{topic} development status",
            f"{topic} clinical trial",
            f"{topic} company pipeline",
        ]

    def _format_all_results(self, results: Dict) -> Dict[str, str]:
        """Format all research results into readable strings."""
        formatted = {}

        if "biomcp" in results and results["biomcp"]:
            biomcp_parts = []
            for key, value in results["biomcp"].items():
                biomcp_parts.append(f"\n### {key}:\n{value}\n")
            formatted["biomcp"] = "\n".join(biomcp_parts)
        else:
            formatted["biomcp"] = "No BioMCP results available"

        if "web" in results and results["web"]:
            web_parts = []
            for query, value in results["web"].items():
                web_parts.append(f"\n### Query: {query}\n{value}\n")
            formatted["web"] = "\n".join(web_parts)
        else:
            formatted["web"] = "No web search results available"

        return formatted


def main(args=None, topic: str = None, reference: str = None):
    """
    Main entry point for Deep Research.

    Args:
        args: Optional arguments (ignored for deep research)
        topic: The research topic/question
        reference: Optional reference papers (not required for deep research)
    """
    if not topic:
        return "Error: Topic is required for deep research"

    web_env = None
    if BROWSER_ENV_AVAILABLE:
        try:
            local_root = os.path.join(os.getcwd(), "workplace_paper")
            os.makedirs(local_root, exist_ok=True)

            container_name = "deep_research_" + str(os.getpid())

            env_config = DockerConfig(
                container_name=container_name,
                workplace_name=DOCKER_WORKPLACE_NAME,
                communication_port=12345,
                local_root=local_root,
            )
            code_env = DockerEnv(env_config)
            code_env.init_container()

            web_env = BrowserEnv(
                browsergym_eval_env=None,
                local_root=env_config.local_root,
                workplace_name=env_config.workplace_name,
            )
        except Exception as e:
            print(f"Warning: Failed to initialize web environment: {e}")
            print("Web search will be skipped.")
    else:
        print("Warning: browsergym not available. Web search will be skipped.")

    async def run_research():
        flow = DeepResearchFlow(
            cache_path=None,
            log_path=None,
            model=CHEEP_MODEL,
            web_env=web_env,
        )
        result = await flow.research(topic=topic)
        return result

    return asyncio.run(run_research())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Deep Research Module")
    parser.add_argument(
        "--topic", type=str, required=True, help="Research topic/question"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-2024-08-06", help="Model to use"
    )

    args = parser.parse_args()

    result = main(topic=args.topic)
    print(result)
