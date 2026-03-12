import json
from typing import Optional, Dict, Any

from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.types import Agent

from research_agent.inno.registry import (
    register_agent,
    get_tools,
)


def case_resolved(research_result: str) -> str:
    """
    Use this function when you have completed the research task with satisfactory results.

    Args:
        research_result: The complete research summary and findings.
    """
    return research_result


def case_not_resolved(suggestions: str) -> str:
    """
    Use this function when the research is incomplete and you need to provide suggestions for improvement.

    Args:
        suggestions: Suggestions for what needs to be done to complete the research.
    """
    return suggestions


DEEP_SURVEY_AGENT_INSTRUCTIONS = """You are a `Deep Survey Agent` specialized in comprehensive research with verification.

## OBJECTIVE
- Perform deep research on the given topic using multiple sources
- Verify research findings for accuracy, completeness, and relevance
- Iteratively fix any issues found during verification
- Provide a comprehensive and accurate research summary

## AVAILABLE TOOLS

### 1. BioMCP Tools (Biomedical Research)
- `biomcp_article_search`: Search biomedical articles
- `biomcp_article_get`: Get article details by PubMed ID
- `biomcp_trial_search`: Search clinical trials
- `biomcp_trial_get`: Get trial details by NCT ID
- `biomcp_gene_search` / `biomcp_gene_get`: Gene information
- `biomcp_drug_search` / `biomcp_drug_get`: Drug information
- `biomcp_disease_search` / `biomcp_disease_get`: Disease information
- `biomcp_variant_search` / `biomcp_variant_get`: Genetic variants
- `biomcp_pathway_search` / `biomcp_pathway_get`: Pathways
- `biomcp_protein_search` / `biomcp_protein_get`: Proteins
- `biomcp_adverse_event_search` / `biomcp_adverse_event_get`: Adverse events
- `biomcp_pgx_search` / `biomcp_pgx_get`: Pharmacogenomics
- `biomcp_gwas_search`: GWAS studies
- `biomcp_phenotype_search`: Phenotypes
- `biomcp_gene_enrich`: Gene-set enrichment analysis

### 2. OpenAlex Search (Academic Literature)
- `openalex_search_papers`: Search papers (query, year_from, year_to, max_results)
- `openalex_search`: Advanced search with filters

### 3. Web Search
- `ddg_search`: DuckDuckGo web search

### 4. Citation Formatting
- `citation_format`: Format citations consistently
- `citation_add`: Add a single citation
- `citation_clear`: Clear all citations
- `citation_get`: Get all stored citations

## WORKFLOW

1. **Analyze** the topic to identify key entities
2. **First, search for related papers** using BioMCP (`biomcp_article_search`) and OpenAlex (`openalex_search_papers`) - these will be your PRIMARY sources for citations
3. **CRITICAL - Verify each article**: After `biomcp_article_search` returns results, you MUST call `biomcp_article_get` for each PMID to verify if the article is relevant to the topic.
4. **Then search** other relevant information using other BioMCP tools and DuckDuckGo
5. **Synthesize** findings into comprehensive summary using the related papers as your main citation sources
6. **Format citations** using `citation_format` tool to store formatted citations in `context_variables['citations']`
7. **Verify** accuracy, completeness, and relevance
8. **Iterate** and fix issues
9. **Return** final verified research summary

## CITATION REQUIREMENTS
- Use papers found in step 2 as PRIMARY citation sources
- Include paper titles, authors, year, and source (PubMed/OpenAlex) in your citations
- Format citations consistently (e.g., [Paper Title, Year, Source])
- Give HIGHER PRIORITY to papers from `biomcp_article_get` and `openalex_search` that are verified as topic-related in your final results

## VERIFICATION CRITERIA
- Accuracy: Are factual claims supported by sources?
- Completeness: All important aspects covered?
- Relevance: Does it address the research topic?

## IMPORTANT
- Use exact topic terms in searches (e.g., "IL1RAP ADC" not just "ADC")
- Search without year limits for latest information
- Always cite your sources using papers from step 2 and 3
- Papers from `biomcp_article_search` and `openalex_search` that are verified as topic-related should be prominently included in your final results 

## MANDATORY TERMINATION

When you have completed the research:
- Call `case_resolved(research_result="your complete research summary here")` 

When the research is incomplete:
- Call `case_not_resolved(suggestions="what needs to be done to complete the research")` 

**IMPORTANT**:
- Pass your complete research summary directlyolved(research_result to `case_res=...)`
- Do NOT call case_resolved with empty or placeholder content
- Include key findings, citations, and conclusions in the research_result"""


@register_agent("get_deep_survey_agent")
def get_deep_survey_agent(model: str, **kwargs) -> Agent:
    file_env: Optional[RequestsMarkdownBrowser] = kwargs.get("file_env")
    if file_env is None:
        raise ValueError("file_env is required")

    tool_names = [
        "biomcp_article_search",
        "biomcp_article_get",
        "biomcp_trial_search",
        "biomcp_trial_get",
        "biomcp_gene_search",
        "biomcp_gene_get",
        "biomcp_drug_search",
        "biomcp_drug_get",
        "biomcp_disease_search",
        "biomcp_disease_get",
        "biomcp_variant_search",
        "biomcp_variant_get",
        "biomcp_pathway_search",
        "biomcp_pathway_get",
        "biomcp_protein_search",
        "biomcp_protein_get",
        "biomcp_adverse_event_search",
        "biomcp_adverse_event_get",
        "biomcp_pgx_search",
        "biomcp_pgx_get",
        "biomcp_gwas_search",
        "biomcp_phenotype_search",
        "biomcp_gene_enrich",
        "openalex_search",
        "openalex_search_papers",
        "ddg_search",
        "citation_format",
        "citation_add",
        "citation_clear",
        "citation_get",
    ]
    tools = get_tools(tool_names, env=file_env, env_wrapper=with_env_file)

    agent = Agent(
        name="Deep Survey Agent",
        model=model,
        instructions=DEEP_SURVEY_AGENT_INSTRUCTIONS,
        functions=tools + [case_resolved, case_not_resolved],
        tool_choice="required",
    )

    return agent
