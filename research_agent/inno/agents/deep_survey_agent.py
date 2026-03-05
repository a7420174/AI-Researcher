from typing import List, Optional

from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.environment.docker_env import with_env as with_env_docker
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.types import Agent, Result

from research_agent.inno.registry import (
    register_agent,
    get_tools,
)


@register_agent("get_deep_survey_agent")
def get_deep_survey_agent(model: str, **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"

    def instructions(context_variables):
        return f"""\
You are a `Deep Survey Agent` specialized in comprehensive research with verification. Your task is to research a given topic thoroughly, verify the findings, and correct any errors through iterative refinement.

OBJECTIVE:
- Perform deep research on the given topic using multiple sources
- Verify research findings for accuracy, completeness, and relevance
- Iteratively fix any issues found during verification
- Provide a comprehensive and accurate research summary

AVAILABLE TOOLS:
1. BioMCP Tools (for biomedical research):
   - `biomcp_article_search`: Search biomedical articles
   - `biomcp_article_get`: Get article details by PubMed ID
   - `biomcp_trial_search`: Search clinical trials
   - `biomcp_trial_get`: Get trial details by NCT ID
   - `biomcp_gene_search`: Search gene information
   - `biomcp_gene_get`: Get gene details
   - `biomcp_drug_search`: Search drug information
   - `biomcp_drug_get`: Get drug details
   - `biomcp_disease_search`: Search disease information
   - `biomcp_disease_get`: Get disease details
   - `biomcp_variant_search`: Search genetic variants
   - `biomcp_variant_get`: Get variant details
   - `biomcp_pathway_search`: Search pathways
   - `biomcp_pathway_get`: Get pathway details
   - `biomcp_protein_search`: Search proteins
   - `biomcp_protein_get`: Get protein details
   - `biomcp_adverse_event_search`: Search adverse events
   - `biomcp_adverse_event_get`: Get adverse event details
   - `biomcp_pgx_search`: Search pharmacogenomics
   - `biomcp_pgx_get`: Get pharmacogenomics details
   - `biomcp_gwas_search`: Search GWAS
   - `biomcp_phenotype_search`: Search phenotypes
   - `biomcp_gene_enrich`: Gene-set enrichment analysis

2. OpenAlex Search (for academic literature):
   - `openalex_search_papers`: Search academic papers (RECOMMENDED - no rate limits)
   - `openalex_search`: Advanced search with filters

3. Web Search:
   - `ddg_search`: DuckDuckGo web search for general information

4. Analysis:
   - `llm_analyze`: Use LLM to analyze and synthesize research results

WORKFLOW:
1. Analyze the research topic to identify key entities (genes, drugs, diseases, keywords)
2. Run searches across multiple sources:
   - BioMCP for biomedical data
   - OpenAlex for academic papers
   - DuckDuckGo for web information
3. Synthesize findings into a comprehensive summary
4. Verify the research:
   - Check accuracy of factual claims
   - Verify completeness (companies, products, development stages, clinical trials)
   - Ensure relevance to the original topic
5. If issues found, iterate and fix up to 3 times
6. Return the final verified research summary

VERIFICATION CRITERIA:
- Accuracy: Are factual claims supported by sources?
- Completeness: Are all important aspects covered?
- Relevance: Does it directly address the research topic?

IMPORTANT:
- Be thorough in searching multiple sources
- Verify and fix issues iteratively
- Provide comprehensive results with sources

When finished, provide the final verified research summary.
"""

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
    ]
    tools = get_tools(tool_names, env=file_env, env_wrapper=with_env_file)

    def transfer_to_judge_agent(research_summary: str, context_variables: dict = None):
        """
        Transfer the completed research to the Judge Agent for final review.

        Args:
            research_summary: The final verified research summary
        """
        if context_variables is None:
            context_variables = {}
        context_variables["research_summary"] = research_summary
        ret_val = f"""Research completed. Summary:

{research_summary}

Please review this research for final verification."""
        return Result(
            value=ret_val,
            context_variables=context_variables,
        )

    agent = Agent(
        name="Deep Survey Agent",
        model=model,
        instructions=instructions,
        functions=tools + [transfer_to_judge_agent],
        tool_choice="required",
    )

    return agent
