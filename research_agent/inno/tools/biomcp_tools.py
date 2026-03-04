import subprocess
import json
import os
import shutil
from typing import Optional, List
from research_agent.inno.registry import register_tool


# Cache for the biomcp binary path
_biomcp_binary_cache = None


def _find_biomcp_binary() -> str:
    """
    Automatically find the biomcp binary.

    Checks in order:
    1. .venv/bin/biomcp relative to current script location
    2. .venv/bin/biomcp relative to current working directory
    3. biomcp in system PATH

    Returns:
        Path to biomcp binary

    Raises:
        FileNotFoundError: If biomcp binary is not found
    """
    global _biomcp_binary_cache

    if _biomcp_binary_cache is not None:
        # Always verify the cached path still exists
        if os.path.isfile(_biomcp_binary_cache) and os.access(
            _biomcp_binary_cache, os.X_OK
        ):
            return _biomcp_binary_cache
        # Cache is invalid, clear it
        _biomcp_binary_cache = None

    # Possible locations to check (in order of preference)
    # 1. .venv/bin/biomcp relative to project root (4 levels up from this file)
    # 2. .venv/bin/biomcp relative to current working directory
    # 3. biomcp in system PATH
    script_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    possible_paths = [
        # Relative to script location (project root)
        os.path.join(script_dir, ".venv", "bin", "biomcp"),
        # Relative to current working directory
        os.path.join(os.getcwd(), ".venv", "bin", "biomcp"),
    ]

    # Check each possible path
    for path in possible_paths:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            _biomcp_binary_cache = path
            return path

    # Fall back to searching in PATH
    biomcp_path = shutil.which("biomcp")
    if biomcp_path:
        _biomcp_binary_cache = biomcp_path
        return biomcp_path

    # If not found, return a placeholder (will fail at runtime)
    _biomcp_binary_cache = (
        "biomcp"  # Use 'biomcp' as fallback, will fail gracefully if not found
    )
    return _biomcp_binary_cache


# Auto-detect biomcp binary path (deferred until actually used)
BIOMCP_BINARY = None


def _get_biomcp_binary() -> str:
    """Lazy initialization of BIOMCP_BINARY"""
    global BIOMCP_BINARY
    if BIOMCP_BINARY is None:
        BIOMCP_BINARY = _find_biomcp_binary()
    return BIOMCP_BINARY


def _run_biomcp_command(args: List[str]) -> str:
    """Run biomcp CLI command and return output."""
    biomcp_bin = _get_biomcp_binary()
    cmd = [biomcp_bin] + args

    # Pass environment variables including API keys
    env = os.environ.copy()

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, env=env
        )
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
    disease: Optional[str] = None,
    keyword: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search biomedical research articles.

    Args:
        gene: Gene name to search for (e.g., BRAF, KRAS)
        disease: Disease name to search for (e.g., melanoma)
        keyword: General keyword to search for
        since: Filter by publication date (e.g., 2024-01-01)
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "article"]

    if gene:
        args.extend(["-g", gene])
    if disease:
        args.extend(["-d", disease])
    if keyword:
        args.extend(["-q", keyword])
    if since:
        args.extend(["--since", since])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_article_get")
def biomcp_article_get(
    id: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve article details by PubMed ID.

    Args:
        id: PubMed ID (PMID)
        section: Optional section (e.g., "fulltext")
    """
    args = ["get", "article", id]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_trial_search")
def biomcp_trial_search(
    condition: Optional[str] = None,
    intervention: Optional[str] = None,
    status: Optional[str] = None,
    source: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search clinical trials.

    Args:
        condition: Medical condition to search for
        intervention: Treatment/intervention to search for
        status: Recruiting status (recruiting, not_recruiting, etc.)
        source: Data source (ctgov, etc.)
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "trial"]

    if condition:
        args.extend(["-c", condition])
    if intervention:
        args.extend(["-i", intervention])
    if status:
        args.extend(["--status", status])
    if source:
        args.extend(["--source", source])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_trial_get")
def biomcp_trial_get(
    nct_id: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve clinical trial details by NCT ID.

    Args:
        nct_id: Clinical trial NCT ID (e.g., NCT04280705)
        section: Optional section (e.g., "eligibility")
    """
    args = ["get", "trial", nct_id]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_variant_search")
def biomcp_variant_search(
    gene: Optional[str] = None,
    hgvsp: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search genetic variants.

    Args:
        gene: Gene symbol (e.g., BRAF)
        hgvsp: Protein notation (e.g., V600E)
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "variant"]

    if gene:
        args.extend(["-g", gene])
    if hgvsp:
        args.extend(["--hgvsp", hgvsp])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_variant_get")
def biomcp_variant_get(
    variant: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve variant details.

    Args:
        variant: Genetic variant identifier (e.g., "BRAF V600E")
        section: Optional section (e.g., "predict")
    """
    args = ["get", "variant", variant]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_search")
def biomcp_gene_search(
    gene: str,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search gene information.

    Args:
        gene: Gene name or symbol to search for (e.g., BRAF, TP53)
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = [
        "search",
        "gene",
        "-q",
        gene,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
    ]

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_get")
def biomcp_gene_get(
    gene: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve gene details.

    Args:
        gene: Gene symbol or ID (e.g., BRAF)
        section: Optional section (e.g., "pathways", "diseases")
    """
    args = ["get", "gene", gene]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_drug_search")
def biomcp_drug_search(
    drug: str,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search drug information.

    Args:
        drug: Drug name to search for
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = [
        "search",
        "drug",
        "-q",
        drug,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
    ]

    return _run_biomcp_command(args)


@register_tool("biomcp_drug_get")
def biomcp_drug_get(
    drug: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve drug details.

    Args:
        drug: Drug name or identifier
        section: Optional section (e.g., "shortage")
    """
    args = ["get", "drug", drug]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_disease_search")
def biomcp_disease_search(
    disease: str,
    source: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search disease information.

    Args:
        disease: Disease name to search for
        source: Data source (e.g., "mondo")
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "disease", "-q", disease]

    if source:
        args.extend(["--source", source])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_disease_get")
def biomcp_disease_get(
    disease: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve disease details.

    Args:
        disease: Disease name or identifier (e.g., "melanoma" or "MONDO:0005105")
        section: Optional section (e.g., "genes", "variants", "pathways")
    """
    args = ["get", "disease", disease]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_gene_enrich")
def biomcp_gene_enrich(
    genes: str,
    limit: int = 10,
) -> str:
    """
    Perform gene-set enrichment analysis.

    Args:
        genes: Comma-separated gene list (e.g., "BRAF,KRAS,NRAS")
        limit: Number of enrichment results to return
    """
    args = ["enrich", genes, "--limit", str(limit)]

    return _run_biomcp_command(args)


@register_tool("biomcp_pathway_search")
def biomcp_pathway_search(
    pathway: str,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search pathway information.

    Args:
        pathway: Pathway keyword to search for
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = [
        "search",
        "pathway",
        "-q",
        pathway,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
    ]

    return _run_biomcp_command(args)


@register_tool("biomcp_pathway_get")
def biomcp_pathway_get(
    pathway_id: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve pathway details.

    Args:
        pathway_id: Pathway ID (e.g., "R-HSA-5673001")
        section: Optional section (e.g., "genes")
    """
    args = ["get", "pathway", pathway_id]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_protein_search")
def biomcp_protein_search(
    protein: str,
    all_species: bool = False,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search protein information.

    Args:
        protein: Protein keyword to search for
        all_species: Search across all species
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "protein", "-q", protein]

    if all_species:
        args.append("--all-species")

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_protein_get")
def biomcp_protein_get(
    protein_id: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve protein details.

    Args:
        protein_id: Protein ID (e.g., "P15056")
        section: Optional section (e.g., "domains", "interactions")
    """
    args = ["get", "protein", protein_id]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_adverse_event_search")
def biomcp_adverse_event_search(
    drug: Optional[str] = None,
    serious: bool = False,
    event_type: Optional[str] = None,
    manufacturer: Optional[str] = None,
    product_code: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search adverse event information.

    Args:
        drug: Drug name to search for
        serious: Filter for serious events only
        event_type: Type of event (e.g., "device")
        manufacturer: Manufacturer name
        product_code: Product code
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "adverse-event"]

    if drug:
        args.extend(["--drug", drug])
    if serious:
        args.append("--serious")
    if event_type:
        args.extend(["--type", event_type])
    if manufacturer:
        args.extend(["--manufacturer", manufacturer])
    if product_code:
        args.extend(["--product-code", product_code])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_adverse_event_get")
def biomcp_adverse_event_get(
    event_id: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve adverse event details.

    Args:
        event_id: Adverse event ID
        section: Optional section (e.g., "reactions", "outcomes", "all")
    """
    args = ["get", "adverse-event", event_id]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_pgx_search")
def biomcp_pgx_search(
    gene: Optional[str] = None,
    drug: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search pharmacogenomics information.

    Args:
        gene: Gene symbol (e.g., CYP2D6)
        drug: Drug name (e.g., warfarin)
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "pgx"]

    if gene:
        args.extend(["-g", gene])
    if drug:
        args.extend(["-d", drug])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_pgx_get")
def biomcp_pgx_get(
    gene: str,
    section: Optional[str] = None,
) -> str:
    """
    Retrieve pharmacogenomics details.

    Args:
        gene: Gene symbol (e.g., CYP2D6)
        section: Optional section (e.g., "recommendations", "frequencies")
    """
    args = ["get", "pgx", gene]
    if section:
        args.append(section)

    return _run_biomcp_command(args)


@register_tool("biomcp_gwas_search")
def biomcp_gwas_search(
    gene: Optional[str] = None,
    trait: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search GWAS information.

    Args:
        gene: Gene symbol
        trait: Phenotype/trait description
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = ["search", "gwas"]

    if gene:
        args.extend(["-g", gene])
    if trait:
        args.extend(["--trait", trait])

    args.extend(["--limit", str(limit), "--offset", str(offset)])

    return _run_biomcp_command(args)


@register_tool("biomcp_phenotype_search")
def biomcp_phenotype_search(
    phenotype: str,
    limit: int = 10,
    offset: int = 0,
) -> str:
    """
    Search phenotype information.

    Args:
        phenotype: Phenotype HPO term (e.g., "HP:0001250")
        limit: Maximum number of results
        offset: Offset for pagination
    """
    args = [
        "search",
        "phenotype",
        phenotype,
        "--limit",
        str(limit),
        "--offset",
        str(offset),
    ]

    return _run_biomcp_command(args)
