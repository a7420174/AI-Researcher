import json
from typing import Optional, Dict, Any

from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.types import Agent

from research_agent.inno.registry import (
    register_agent,
    get_tools,
)


def case_resolved(
    context_variables: dict,
    fully_correct: bool,
    suggestion: Optional[Dict[str, str]] = None,
) -> str:
    """
    Use this function when you have finished the task.

    Args:
       context_variables: Must contain "final_research" key with the research summary
       fully_correct: whether the implementation/response is fully correct
       suggestion: dict {key_point: suggestion}. If fully_correct, set to None
    """
    suggestion_dict: Dict[str, Any] = {
        "fully_correct": fully_correct,
    }

    if suggestion:
        suggestion_dict["suggestion"] = suggestion

    context_variables["suggestion_dict"] = suggestion_dict

    return "[DONE]"


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
**You MUST call the `case_resolved` function when you have finished the task.**

CRITICAL: Before calling `case_resolved`, you MUST store your COMPLETE research summary in `context_variables["final_research"]`.

Example:
```
context_variables["final_research"] = "Your full research summary here with all findings and citations"
case_resolved(fully_correct=True)
```

**IMPORTANT**: 
- The content in `context_variables["final_research"]` will be used as the final output
- Do NOT call `case_resolved` before storing the research summary
- Do NOT just say "Let me compile..." - actually store the full research

Then call:
- `case_resolved(fully_correct=True)` when research is complete
- `case_resolved(fully_correct=False, suggestion={...})` if more work is needed

**The task is NOT complete until you call `case_resolved`.**"""


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
        functions=tools + [case_resolved],
        tool_choice="required",
    )

    return agent
