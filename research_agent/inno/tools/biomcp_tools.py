import subprocess
import json
import asyncio
from typing import Optional, List
from research_agent.inno.registry import register_tool


@register_tool("biomcp_think")
def biomcp_think(
    thought: str,
    thoughtNumber: int = 1,
    totalThoughts: int = 1,
    nextThoughtNeeded: bool = False,
    isRevision: bool = False,
    revisesThought: Optional[int] = None,
) -> str:
    """
    Sequential thinking tool for complex reasoning about biomedical queries.

    Use this BEFORE searching to plan your search strategy:
    - Break down the query into components
    - Identify what databases/entities to search
    - Determine search terms and filters

    Args:
        thought: Your current thinking step - be detailed and thorough
        thoughtNumber: Current thought number (start at 1)
        totalThoughts: Best estimate of total thoughts needed
        nextThoughtNeeded: True if more thinking needed, False when done
        isRevision: True when correcting/improving a previous thought
        revisesThought: The thought number being revised (if isRevision=True)
    """
    from biomcp.thinking.sequential import _sequential_thinking

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            _sequential_thinking(
                thought=thought,
                nextThoughtNeeded=nextThoughtNeeded,
                thoughtNumber=thoughtNumber,
                totalThoughts=totalThoughts,
                isRevision=isRevision,
                revisesThought=revisesThought,
                branchFromThought=None,
                needsMoreThoughts=None,
            )
        )
        loop.close()
        return result
    except Exception as e:
        return f"Error in biomcp_think: {str(e)}"


def _run_biomcp_command(args: List[str]) -> str:
    """Run biomcp CLI command and return output."""
    cmd = ["biomcp"] + args
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            return f"Error: {result.stderr}"
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error: {str(e)}"


@register_tool("biomcp_article_search")
def biomcp_article_search(
    gene: Optional[str] = None,
    variant: Optional[str] = None,
    disease: Optional[str] = None,
    chemical: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 10,
    page: int = 1,
) -> str:
    """
    Search biomedical research articles from PubMed/PubTator3.

    Args:
        gene: Gene name to search for (e.g., BRAF, KRAS)
        variant: Genetic variant to search for (e.g., BRAF V600E)
        disease: Disease name to search for (e.g., melanoma, lung cancer)
        chemical: Chemical/drug name to search for
        keyword: General keyword to search for
        limit: Maximum number of results (1-100)
        page: Page number for pagination
    """
    args = ["article", "search"]

    if gene:
        args.extend(["--gene", gene])
    if variant:
        args.extend(["--variant", variant])
    if disease:
        args.extend(["--disease", disease])
    if chemical:
        args.extend(["--chemical", chemical])
    if keyword:
        args.extend(["--keyword", keyword])

    args.extend(["--limit", str(limit), "--page", str(page)])

    return _run_biomcp_command(args)


@register_tool("biomcp_article_get")
def biomcp_article_get(
    id: str,
    full: bool = False,
) -> str:
    """
    Retrieve article details by PubMed ID or DOI.

    Args:
        id: PubMed ID (PMID) or DOI
        full: Include full abstract text
    """
    args = ["article", "get", id]
    if full:
        args.append("--full")

    return _run_biomcp_command(args)


@register_tool("biomcp_trial_search")
def biomcp_trial_search(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    lead_sponsor: Optional[str] = None,
    term: Optional[str] = None,
    nct_id: Optional[str] = None,
    status: str = "open",
    phase: Optional[str] = None,
    age_group: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Search clinical trials from ClinicalTrials.gov.

    Args:
        condition: Medical condition to search for
        intervention: Treatment/intervention to search for
        lead_sponsor: Lead sponsor organization name
        term: General search terms
        nct_id: Clinical trial NCT ID
        status: Recruiting status (open/closed/any)
        phase: Trial phase (early_phase1, phase1, phase2, phase3, phase4)
        age_group: Age group filter (child/adult/senior/all)
        limit: Number of results to return
    """
    args = ["trial", "search"]

    if condition:
        args.extend(["--condition", condition])
    if intervention:
        args.extend(["--intervention", intervention])
    if lead_sponsor:
        args.extend(["--lead-sponsor", lead_sponsor])
    if term:
        args.extend(["--term", term])
    if nct_id:
        args.extend(["--nct-id", nct_id])

    args.extend(["--status", status])

    if phase:
        args.extend(["--phase", phase])
    if age_group:
        args.extend(["--age-group", age_group])

    args.extend(["--page-size", str(limit)])

    return _run_biomcp_command(args)


@register_tool("biomcp_trial_get")
def biomcp_trial_get(
    nct_id: str,
    json: bool = True,
) -> str:
    """
    Retrieve clinical trial details by NCT ID.

    Args:
        nct_id: Clinical trial NCT ID (e.g., NCT04280705)
        json: Return in JSON format
    """
    args = ["trial", "get", nct_id]
    if json:
        args.append("--json")

    return _run_biomcp_command(args)


@register_tool("biomcp_variant_search")
def biomcp_variant_search(
    variant: str,
    gene: Optional[str] = None,
    hgvsp: Optional[str] = None,
    significance: Optional[str] = None,
    limit: int = 10,
) -> str:
    """
    Search genetic variants from MyVariant.info.

    Args:
        variant: Genetic variant to search for (e.g., BRAF V600E)
        gene: Gene symbol (e.g., BRAF)
        hgvsp: Protein notation (e.g., p.Val600Glu)
        significance: Clinical significance (pathogenic, likely_pathogenic, etc.)
        limit: Number of results to return
    """
    args = ["variant", "search"]

    if variant:
        args.extend(["--gene", variant])
    if gene:
        args.extend(["--gene", gene])
    if hgvsp:
        args.extend(["--hgvsp", hgvsp])
    if significance:
        args.extend(["--significance", significance])

    args.extend(["--size", str(limit)])

    return _run_biomcp_command(args)


@register_tool("biomcp_variant_get")
def biomcp_variant_get(
    variant: str,
    source: Optional[str] = None,
) -> str:
    """
    Retrieve variant details from specific source.

    Args:
        variant: Genetic variant identifier
        source: Data source (clinvar, gnomad, myvariant, civic, oncokb)
    """
    args = ["variant", "get", variant]
    if source:
        args.extend(["--source", source])

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_search")
def biomcp_gene_search(
    gene: str,
    limit: int = 10,
) -> str:
    """
    Search gene information from MyGene.info.

    Args:
        gene: Gene name or symbol to search for (e.g., BRAF, TP53)
        limit: Number of results to return
    """
    args = ["gene", "search", gene, "--page-size", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_get")
def biomcp_gene_get(
    gene: str,
    species: Optional[str] = None,
) -> str:
    """
    Retrieve gene details from MyGene.info.

    Args:
        gene: Gene symbol or ID
        species: Species filter (human, mouse, rat)
    """
    args = ["gene", "get", gene]
    if species:
        args.extend(["--species", species])

    return _run_biomcp_command(args)


@register_tool("biomcp_drug_search")
def biomcp_drug_search(
    drug: str,
    limit: int = 10,
) -> str:
    """
    Search drug information from MyChem.info.

    Args:
        drug: Drug name to search for
        limit: Number of results to return
    """
    args = ["drug", "search", drug, "--page-size", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_drug_get")
def biomcp_drug_get(
    drug: str,
) -> str:
    """
    Retrieve drug details from MyChem.info.

    Args:
        drug: Drug name or identifier
    """
    args = ["drug", "get", drug]

    return _run_biomcp_command(args)


@register_tool("biomcp_disease_search")
def biomcp_disease_search(
    disease: str,
    limit: int = 10,
) -> str:
    """
    Search disease information from BioThings API.

    Args:
        disease: Disease name to search for
        limit: Number of results to return
    """
    args = ["disease", "search", disease, "--page-size", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_disease_get")
def biomcp_disease_get(
    disease: str,
) -> str:
    """
    Retrieve disease details from BioThings API.

    Args:
        disease: Disease name or identifier
    """
    args = ["disease", "get", disease]

    return _run_biomcp_command(args)


@register_tool("biomcp_intervention_search")
def biomcp_intervention_search(
    intervention: str,
    limit: int = 10,
) -> str:
    """
    Search intervention information from NCI CTS API.

    Args:
        intervention: Intervention/treatment name to search for
        limit: Number of results to return
    """
    args = ["intervention", "search", intervention, "--page-size", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_biomarker_search")
def biomcp_biomarker_search(
    biomarker: str,
    limit: int = 10,
) -> str:
    """
    Search biomarker information used in clinical trials.

    Args:
        biomarker: Biomarker name to search for (e.g., PD-L1, HER2)
        limit: Number of results to return
    """
    args = ["biomarker", "search", biomarker, "--page-size", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_enrich")
def biomcp_gene_enrich(
    genes: str,
    limit: int = 10,
) -> str:
    """
    Perform gene-set enrichment analysis.

    Args:
        genes: Comma-separated gene list (e.g., BRAF,KRAS,NRAS)
        limit: Number of enrichment results to return
    """
    args = ["gene", "enrich", genes, "--page-size", str(limit)]

    return _run_biomcp_command(args)
