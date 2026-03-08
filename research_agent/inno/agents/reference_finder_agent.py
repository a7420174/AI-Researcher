from typing import List

from research_agent.inno.types import Agent, Result

from research_agent.inno.registry import (
    register_agent,
    get_tools,
)


@register_agent("get_reference_finder_agent")
def get_reference_finder_agent(model: str, **kwargs):
    tool_names = [
        "openalex_search_papers",
        "biomcp_article_search",
    ]
    tool_list = get_tools(tool_names)

    def instructions(context_variables):
        return """\
You are a `Reference Finder Agent` specialized in finding methodology papers.

OBJECTIVE:
- Find methodology papers (papers describing METHODS, ALGORITHMS, COMPUTATIONAL APPROACHES) related to the given research topic

AVAILABLE TOOLS:
1. openalex_search_papers: Search academic papers from OpenAlex
   - Parameters: query (str), max_results (int, default 10)

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:
1. Make ONLY ONE tool call with max_results=5
2. Query strategy:
   - Simplify the topic to main keywords (e.g., "UPK1B cancer")
   - DO NOT add "method", "algorithm" to query - these return no results
3. DO NOT make additional tool calls
4. After getting results, select methodology papers:
   - Include: computational methods, algorithms, ML models, analysis pipelines, deep learning, new approaches
   - Exclude: Editorial, Review, Abstract, biomarker, prognostic, diagnostic, case report
5. Return selected titles (max 5), one per line, no numbering
6. If most results are non-methodology, return up to 2 that are most relevant
7. If no relevant papers found, return: No methodology papers found"""

    return Agent(
        name="Reference Finder Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="auto",
    )
