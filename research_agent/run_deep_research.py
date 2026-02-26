"""
Deep Research Module - Docker-independent web-based research
This module performs comprehensive research using search tools, BioMCP, OpenAlex, and LLM.
No Docker dependency required.
"""

import os
import asyncio
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

from research_agent.inno.registry import get_tool
from research_agent.constant import COMPLETION_MODEL, CHEEP_MODEL

# Import tools to register them
from research_agent.inno.tools import biomcp_tools
from research_agent.inno.tools import openalex_tools
from research_agent.inno.tools import web_tools


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

    async def research(
        self,
        topic: str,
        max_search_results: int = 10,
    ) -> str:
        """
        Perform deep research on a given topic using search tools, BioMCP, OpenAlex, and LLM.

        Args:
            topic: The research topic/question
            max_search_results: Maximum number of search results to gather

        Returns:
            Comprehensive research findings
        """
        topic_analysis = await self._analyze_topic(topic)

        biomcp_results = await self._run_biomcp_searches(topic, topic_analysis)

        openalex_results = await self._run_openalex_searches(topic, topic_analysis)

        ddg_results = await self._run_ddg_searches(topic, topic_analysis)

        all_results = {
            "biomcp": biomcp_results,
            "openalex": openalex_results,
            "ddg": ddg_results,
        }

        research_context = self._format_all_results(all_results)

        synthesis_query = f"""You are a research assistant. Based on the following research results, provide a comprehensive research summary on the topic:

Topic: {topic}

{"=" * 60}
BIOMEDICAL DATABASE RESULTS (BioMCP):
{"=" * 60}
{research_context.get("biomcp", "No BioMCP results")}

{"=" * 60}
OPENALEX SEARCH RESULTS (Academic Papers):
{"=" * 60}
{research_context.get("openalex", "No OpenAlex results")}

{"=" * 60}
WEB SEARCH RESULTS (DuckDuckGo):
{"=" * 60}
{research_context.get("ddg", "No web search results")}

Please provide:
1. Executive Summary
2. Key findings and information from biomedical databases
3. Key findings from academic literature (OpenAlex)
4. Key findings from web searches
5. Comparison and synthesis of all sources
6. Relevant details (companies, products, development stages, indications, etc.)
7. Sources and references
8. Any gaps or areas needing further research

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

        biomcp_tools_list = []

        if analysis["genes"]:
            gene = analysis["genes"][0]
            biomcp_tools_list.extend(
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
                biomcp_tools_list.append(
                    (
                        "biomcp_article_search",
                        {"gene": gene, "disease": disease, "limit": 10},
                    )
                )
                biomcp_tools_list.append(
                    (
                        "biomcp_trial_search",
                        {"condition": disease, "intervention": gene, "limit": 10},
                    )
                )
            else:
                biomcp_tools_list.append(
                    ("biomcp_article_search", {"disease": disease, "limit": 10})
                )
                biomcp_tools_list.append(
                    ("biomcp_trial_search", {"condition": disease, "limit": 10})
                )

        if analysis.get("keywords"):
            for keyword in analysis["keywords"]:
                biomcp_tools_list.append(
                    ("biomcp_article_search", {"keyword": keyword, "limit": 5})
                )

        if not biomcp_tools_list and analysis["is_biomedical"]:
            main_keyword = (
                analysis["genes"][0] if analysis["genes"] else topic.split()[0]
            )
            biomcp_tools_list.append(
                ("biomcp_article_search", {"keyword": main_keyword, "limit": 5})
            )

        for tool_name, params in biomcp_tools_list:
            try:
                tool = get_tool(tool_name)
                result = tool(**params)
                results[
                    f"{tool_name}_{params.get('gene', params.get('keyword', params.get('disease', 'default')))}"
                ] = result
            except Exception as e:
                results[tool_name] = f"Error: {str(e)}"

        return results

    async def _run_openalex_searches(
        self, topic: str, analysis: Dict
    ) -> Dict[str, str]:
        """Run OpenAlex searches for academic papers."""
        results = {}

        search_queries = await self._generate_search_queries(topic, analysis)

        try:
            search_tool = get_tool("openalex_search_papers")
        except Exception as e:
            for query in search_queries:
                results[query] = f"Error: OpenAlex tools not available: {e}"
            return results

        for query in search_queries:
            try:
                result = search_tool(query=query, max_results=20)
                results[query] = result
            except Exception as e:
                results[query] = f"Error: {str(e)}"

        return results

    async def _run_ddg_searches(self, topic: str, analysis: Dict) -> Dict[str, str]:
        """Run DuckDuckGo web searches."""
        results = {}

        search_queries = await self._generate_search_queries(topic, analysis)

        try:
            search_tool = get_tool("ddg_search")
        except Exception as e:
            for query in search_queries:
                results[query] = f"Error: ddg_search not available: {e}"
            return results

        for query in search_queries:
            try:
                result = search_tool(query=query, max_results=10)
                results[query] = result
            except Exception as e:
                results[query] = f"Error: {str(e)}"

        return results

    async def _generate_search_queries(
        self, topic: str, analysis: Dict = None
    ) -> List[str]:
        """Generate dynamic search queries based on the topic using LLM."""

        query_prompt = f"""Generate effective search queries for the following research question.

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
4. Recent research
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

        if "openalex" in results and results["openalex"]:
            openalex_parts = []
            for key, value in results["openalex"].items():
                openalex_parts.append(f"\n### Query: {key}:\n{value}\n")
            formatted["openalex"] = "\n".join(openalex_parts)
        else:
            formatted["openalex"] = "No OpenAlex results available"

        if "ddg" in results and results["ddg"]:
            ddg_parts = []
            for key, value in results["ddg"].items():
                ddg_parts.append(f"\n### Query: {key}:\n{value}\n")
            formatted["ddg"] = "\n".join(ddg_parts)
        else:
            formatted["ddg"] = "No DuckDuckGo results available"

        return formatted


def main(topic: str, reference: str = None, use_docker: bool = None):
    """
    Main entry point for Deep Research (Docker-independent).

    Args:
        topic: The research topic/question
        reference: Optional reference papers (not used in Deep Research mode)
        use_docker: Whether to use Docker (not used - Deep Research is Docker-independent)
    """
    if not topic:
        return "Error: Topic is required for deep research"

    async def run_research():
        flow = DeepResearchFlow(
            cache_path=None,
            log_path=None,
            model=CHEEP_MODEL,
        )
        result = await flow.research(topic=topic)
        return result

    return asyncio.run(run_research())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deep Research Module (Docker-independent)"
    )
    parser.add_argument(
        "--topic", type=str, required=True, help="Research topic/question"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model to use (default: CHEEP_MODEL)"
    )

    args = parser.parse_args()

    result = main(topic=args.topic)
    print(result)
