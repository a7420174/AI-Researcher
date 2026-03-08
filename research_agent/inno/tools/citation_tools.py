from typing import List, Dict, Any, Optional
from research_agent.inno.registry import register_tool
from research_agent.inno.types import Result


@register_tool("citation_format")
def citation_format(
    context_variables: dict,
    citations: List[Dict[str, Any]],
    format_style: str = "[{title}, {year}, {source}]"
) -> Result:
    """
    Format citations consistently and store them in context_variables.
    
    This tool formats a list of citation information into a consistent format
    and stores the result in context_variables['citations'] for later use.

    Args:
        context_variables: The context variables dictionary (automatically passed).
        citations: A list of citation dictionaries. Each dictionary should contain:
            - title: The paper title (required)
            - year: Publication year (required)
            - source: Source of the citation (e.g., "PubMed", "OpenAlex", "arXiv") (required)
            - authors: List of author names (optional)
            - url: URL to the paper (optional)
            - doi: DOI of the paper (optional)
            - pmid: PubMed ID (optional)
        format_style: The format string for citations. Available placeholders:
            - {title}: Paper title
            - {year}: Publication year
            - {source}: Citation source
            - {authors}: Authors (formatted as "FirstAuthor et al." if many)
            - {url}: URL to the paper
            - {doi}: DOI of the paper
            Default: "[{title}, {year}, {source}]"

    Returns:
        A Result object containing:
        - value: Formatted citations as a string
        - context_variables: Updated with 'citations' key containing formatted citations

    Example:
        Input citations:
        [
            {"title": "IL1RAP ADC", "year": "2023", "source": "PubMed", "authors": ["Smith J", "Doe J"]},
            {"title": "Cancer Therapy", "year": "2022", "source": "OpenAlex", "authors": ["Johnson A"]}
        ]
        
        Output (context_variables['citations']):
        ["[IL1RAP ADC, 2023, PubMed]", "[Cancer Therapy, 2022, OpenAlex]"]
    """
    if not citations:
        context_variables["citations"] = []
        return Result(
            value="No citations provided.",
            context_variables=context_variables
        )
    
    formatted_citations = []
    
    for citation in citations:
        # Extract required fields
        title = citation.get("title", "Unknown Title")
        year = citation.get("year", "n.d.")
        source = citation.get("source", "Unknown Source")
        
        # Extract optional fields
        authors = citation.get("authors", [])
        url = citation.get("url", "")
        doi = citation.get("doi", "")
        pmid = citation.get("pmid", "")
        
        # Format authors
        if authors:
            if len(authors) == 1:
                authors_str = authors[0]
            elif len(authors) == 2:
                authors_str = f"{authors[0]} and {authors[1]}"
            else:
                authors_str = f"{authors[0]} et al."
        else:
            authors_str = "Unknown Authors"
        
        # Build the formatted citation using the format_style
        try:
            formatted = format_style.format(
                title=title,
                year=year,
                source=source,
                authors=authors_str,
                url=url,
                doi=doi,
                pmid=pmid
            )
        except KeyError as e:
            # If format string has invalid placeholder, use default
            formatted = f"[{title}, {year}, {source}]"
        
        formatted_citations.append(formatted)
    
    # Store in context_variables
    context_variables["citations"] = formatted_citations
    
    # Build response string
    response_lines = [f"Formatted {len(formatted_citations)} citations:"]
    for i, cit in enumerate(formatted_citations, 1):
        response_lines.append(f"{i}. {cit}")
    
    return Result(
        value="\n".join(response_lines),
        context_variables=context_variables
    )


@register_tool("citation_add")
def citation_add(
    context_variables: dict,
    title: str,
    year: str,
    source: str,
    authors: Optional[List[str]] = None,
    url: str = "",
    doi: str = "",
    pmid: str = ""
) -> Result:
    """
    Add a single citation to the citations list in context_variables.
    
    This tool adds one citation to the existing list of citations in
    context_variables['citations']. If no citations exist yet, it creates a new list.

    Args:
        context_variables: The context variables dictionary (automatically passed).
        title: The paper title (required).
        year: Publication year (required).
        source: Source of the citation (e.g., "PubMed", "OpenAlex", "arXiv") (required).
        authors: List of author names (optional).
        url: URL to the paper (optional).
        doi: DOI of the paper (optional).
        pmid: PubMed ID (optional).

    Returns:
        A Result object containing:
        - value: Confirmation message with the added citation
        - context_variables: Updated with 'citations' key

    Example:
        Input: title="IL1RAP ADC", year="2023", source="PubMed", authors=["Smith J"]
        
        Output (context_variables['citations']):
        ["[IL1RAP ADC, 2023, PubMed]"]
    """
    # Get existing citations or initialize empty list
    existing_citations = context_variables.get("citations", [])
    
    # Format authors
    authors_str = ""
    if authors:
        if len(authors) == 1:
            authors_str = authors[0]
        elif len(authors) == 2:
            authors_str = f"{authors[0]} and {authors[1]}"
        else:
            authors_str = f"{authors[0]} et al."
    
    # Format the citation
    formatted = f"[{title}, {year}, {source}]"
    
    # Add to existing list
    existing_citations.append(formatted)
    
    # Update context_variables
    context_variables["citations"] = existing_citations
    
    return Result(
        value=f"Added citation: {formatted}\nTotal citations: {len(existing_citations)}",
        context_variables=context_variables
    )


@register_tool("citation_clear")
def citation_clear(
    context_variables: dict
) -> Result:
    """
    Clear all citations from context_variables.
    
    This tool removes all citations from context_variables['citations'],
    resetting it to an empty list.

    Args:
        context_variables: The context variables dictionary (automatically passed).

    Returns:
        A Result object containing:
        - value: Confirmation message
        - context_variables: Updated with empty 'citations' list

    Example:
        Before: context_variables['citations'] = ["[Paper1, 2023, PubMed]", "[Paper2, 2022, OpenAlex]"]
        After: context_variables['citations'] = []
    """
    context_variables["citations"] = []
    
    return Result(
        value="All citations have been cleared.",
        context_variables=context_variables
    )


@register_tool("citation_get")
def citation_get(
    context_variables: dict
) -> str:
    """
    Get all citations from context_variables.
    
    This tool retrieves all citations stored in context_variables['citations']
    and returns them as a formatted string.

    Args:
        context_variables: The context variables dictionary (automatically passed).

    Returns:
        A string containing all citations, or a message if no citations exist.

    Example:
        Input: context_variables['citations'] = ["[Paper1, 2023, PubMed]", "[Paper2, 2022, OpenAlex]"]
        
        Output:
        "Stored Citations:
        1. [Paper1, 2023, PubMed]
        2. [Paper2, 2022, OpenAlex]"
    """
    citations = context_variables.get("citations", [])
    
    if not citations:
        return "No citations stored."
    
    response_lines = [f"Stored Citations ({len(citations)}):"]
    for i, cit in enumerate(citations, 1):
        response_lines.append(f"{i}. {cit}")
    
    return "\n".join(response_lines)

