from typing import List, Callable, Optional

from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser
from research_agent.inno.environment.docker_env import with_env as with_env_docker
from research_agent.inno.environment.docker_env import DockerConfig, DockerEnv
from research_agent.inno.types import Agent, Result

from research_agent.inno.registry import (
    register_agent,  # 에이전트 팩토리 등록
    get_tools,  # 툴 이름 목록 → (env 자동 주입) 함수 리스트
    get_agent_factory,  # 다른 에이전트 팩토리 조회
)

from research_agent.inno.tools import biomcp_tools
from research_agent.inno.tools import openalex_tools  # OpenAlex tools


# --------------------------------------------------------------------
# 1) Paper Survey Agent  (레지스트리 등록)
#    - 논문에서 학술 정의/수식/핵심 개념 추출
# --------------------------------------------------------------------
@register_agent("get_paper_survey_agent")
def get_paper_survey_agent(model: str, **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"

    def instructions(context_variables):
        return f"""\
You are a `Paper Survey Agent` specialized in analyzing academic papers and biomedical databases. Your task is to extract and analyze specific academic concepts from research papers located in `{file_env.docker_workplace}/papers/` or from biomedical databases via BioMCP.

OBJECTIVE:
- Analyze the provided academic definition
- Extract relevant mathematical formulas and theoretical foundations
- For biomedical concepts (genes, drugs, diseases, variants), search BioMCP databases when local papers are not available
- Prepare comprehensive notes for the `Code Survey Agent`

AVAILABLE TOOLS:
1. Paper Navigation (use when local papers are available):
   - `open_local_file`: Open and read paper files
   - `page_up_markdown` / `page_down_markdown`: Navigate through pages
   - `find_on_page_ctrl_f` / `find_next`: Search specific content
   - `question_answer_on_whole_page`: Ask specific questions about the paper

3. BioMCP Tools (use for biomedical research):
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

4. OpenAlex Search (PRIORITY for academic literature - no rate limits, better abstracts):
   - `openalex_search_papers`: Search academic papers from OpenAlex (RECOMMENDED for literature review)
     Parameters: query, year_from, year_to, primary_source, max_results
   - `openalex_search`: Advanced search with filters
     Parameters: query, filter (e.g., "publication_year:>=2020"), per_page, page, max_items, sort
   - Use OpenAlex instead of web search when possible for better results

WORKFLOW:
1. First analyze the academic definition to identify key concepts:
   - Break down the academic definition into components
   - Identify which databases/entities to search (gene, drug, disease, variant, article, trial)
   - Determine optimal search terms and filters
2. Then try to find relevant papers in local files (`{file_env.docker_workplace}/papers/`)
3. For academic literature search (especially for papers, reviews, citations):
   - Use `openalex_search_papers` FIRST - it has no rate limits and provides better abstracts
   - OpenAlex is recommended over web search for academic literature
4. For biomedical concepts (genes, drugs, diseases, variants, clinical trials):
   - Use BioMCP tools to search PubMed, ClinicalTrials.gov, MyGene.info, MyChem.info, etc.
5. Extract:
   - Formal definitions
   - Mathematical formulas (for ML/AI concepts)
   - Biomedical information (for gene, drug, disease, variant concepts)
   - Key theoretical components
6. Document your findings and transfer your findings to the `Code Survey Agent` using the `transfer_to_code_survey_agent` function.

REQUIREMENTS:
- Be thorough in your analysis
- Focus on mathematical precision for ML/AI concepts
- For biomedical concepts, prioritize BioMCP searches when local papers are unavailable
- Ensure all extracted information is directly relevant to the given academic definition
- Provide clear and structured notes that can be effectively used by the Code Survey Agent

Remember: Your analysis forms the theoretical foundation for the subsequent code implementation phase.
"""

    paper_tool_names = [
        "open_local_file",
        "page_up_markdown",
        "page_down_markdown",
        "find_on_page_ctrl_f",
        "find_next",
        "question_answer_on_whole_page",
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
    tool_list = get_tools(paper_tool_names, env=file_env, env_wrapper=with_env_file)

    return Agent(
        name="Paper Survey Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="required",
        parallel_tool_calls=False,
    )


# --------------------------------------------------------------------
# 3) Code Survey Agent  (레지스트리 등록)
#    - 아이디어/수식을 코드 구현으로 매핑
# --------------------------------------------------------------------
@register_agent("get_code_survey_agent")
def get_code_survey_agent(model: str, **kwargs):
    code_env: DockerEnv = kwargs.get("code_env", None)
    assert code_env is not None, "code_env is required"

    def instructions(context_variables):
        return f"""\
You are a `Code Survey Agent` specialized in analyzing code implementations of academic concepts. Your task is to examine codebases and match theoretical concepts with their practical implementations.

OBJECTIVE:
- Analyze codebases from reference papers in `/{code_env.workplace_name}/`
- Map academic definitions and mathematical formulas to their code implementations
- Create comprehensive implementation notes

AVAILABLE TOOLS:
1. Code Navigation:
   - `gen_code_tree_structure`: Generate repository structure overview
   - `read_file`: Access and read specific files
   - `terminal_page_down`: Scroll the viewport DOWN one page-length in the current terminal. Use this function when output of the tool is too long and you want to scroll down to see the next content.
   - `terminal_page_up`: Scroll the viewport UP one page-length in the current terminal. Use this function when output of the tool is too long and you want to scroll up to see the previous content.
   - `terminal_page_to`: Move the viewport to the specific page index. Use this function when the terminal output contains a progress bar or output of generating directory structure when there are many datasets in the directory, you can use this function to move the viewport to the end of terminal where the meaningful content is.

2. Documentation:
   - `transfer_back_to_survey_agent`: Document findings and merge with `Paper Survey Agent`'s notes

WORKFLOW:
1. Review provided academic definitions and formulas from `Paper Survey Agent`
2. Generate and analyze codebase structure
3. Locate relevant implementation files
4. Extract and document:
   - Code implementations
   - Implementation details
   - Key functions and classes
5. Merge findings with `Paper Survey Agent`'s notes and transfer complete documentation back to `Survey Agent`using the `transfer_back_to_survey_agent` function

REQUIREMENTS:
- Ensure code examples directly correspond to theoretical concepts
- Focus on critical implementation details
- Document any important variations or optimizations
- Provide clear connections between theory and implementation
"""

    code_tool_names = [
        "gen_code_tree_structure",
        "read_file",
        "terminal_page_down",
        "terminal_page_up",
        "terminal_page_to",
        "list_files",
    ]
    tool_list = get_tools(code_tool_names, env=code_env, env_wrapper=with_env_docker)

    return Agent(
        name="Code Survey Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="required",
        parallel_tool_calls=False,
    )


# --------------------------------------------------------------------
# 공통 유틸: Survey 단계에서 메모 병합 (Result 반환)
# --------------------------------------------------------------------
def case_resolved(context_variables: dict = None):
    """
    After you have taken enough notes for the innovation, you should use this function
    to merge the notes for the further innovation.
    """
    # Handle None context_variables
    if context_variables is None:
        context_variables = {}

    # Ensure notes exist and handle missing keys gracefully
    if not context_variables.get("notes"):
        merge_notes = "No notes available."
    else:
        merge_notes = "\n".join(
            [
                f"## {note.get('definition', 'N/A')}\n"
                f"* The math formula is:\n{note.get('math_formula', 'N/A')}\n"
                f"* * The code implementation is:\n{note.get('code_implementation', 'N/A')}\n"
                f"* Reference papers are:\n{note.get('reference_papers', 'N/A')}\n"
                f"* Reference codebases are:\n{note.get('reference_codebases', 'N/A')}"
                for note in context_variables["notes"]
            ]
        )
    ret_val = f"""\
I have merged the notes for the innovation.
The notes are as follows:
{merge_notes}
"""
    return Result(
        value=ret_val,
        context_variables=context_variables,
    )


# --------------------------------------------------------------------
# 4) Survey Orchestrator Agent  (레지스트리 등록)
#    - Paper/Code Agent를 레지스트리에서 로딩하여 연결
# --------------------------------------------------------------------
@register_agent("get_survey_agent")
def get_survey_agent(model: str = "gpt-4o", **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"
    code_env: DockerEnv = kwargs.get("code_env", None)
    assert code_env is not None, "code_env is required"

    def instructions(context_variables):
        return """\
1. INPUT ANALYSIS
- You will receive a list of research papers and their corresponding codebases
- You will also receive specific innovative ideas that need to be implemented

2. ATOMIC DEFINITION BREAKDOWN
- Break down the innovative ideas into atomic academic definitions
- Each atomic definition should be implementable, mathematically grounded, and traceable to specific papers

3. KEY CONCEPT IDENTIFICATION
- For each atomic definition:
  a. Send it to `Paper Survey Agent` via `transfer_to_paper_survey_agent`
  b. `Paper Survey Agent` extracts definitions & formulas
  c. `Paper Survey Agent` forwards to `Code Survey Agent` via `transfer_to_code_survey_agent`
  d. `Code Survey Agent` extracts corresponding code implementations
  e. `Code Survey Agent` forwards all findings back via `transfer_back_to_survey_agent`
  f. `Survey Agent` aggregates the notes

4. ITERATIVE PROCESS
- Continue until ALL atomic definitions have been covered
- Do not conclude until all necessary concepts are examined

5. FINAL COMPILATION
- Use `case_resolved` to merge all collected notes into a final summary
- Ensure the final output is well-structured and comprehensive
"""

    # 레지스트리에서 다른 에이전트 팩토리를 안전하게 로드
    get_paper_survey_agent_factory = get_agent_factory("get_paper_survey_agent")
    paper_survey_agent = get_paper_survey_agent_factory(model, file_env=file_env)

    get_code_survey_agent_factory = get_agent_factory("get_code_survey_agent")
    code_survey_agent = get_code_survey_agent_factory(model, code_env=code_env)

    survey_agent = Agent(
        name="Survey Agent",
        model=model,
        instructions=instructions,
        tool_choice="required",
        parallel_tool_calls=False,
    )

    # ----- Orchestration Tools (Survey Agent ↔ Paper/Code Agent) -----
    def transfer_back_to_survey_agent(
        academic_definition: str,
        code_implementation: str,
        reference_codebases: List[str],
        context_variables: dict = None,
    ):
        """
        After carefully reading the related paper(s), understanding the academic definition (esp. math formula),
        and reviewing the corresponding code implementation, record structured notes for subsequent innovation.
        Args:
            academic_definition: the academic definition to be explored. It should be a single, atomic academic concept with a few words.
            code_implementation: the code implementation of the academic definition. [IMPORTANT] It should be as complete as possible and it should be the real code.
            reference_codebases: the list of reference codebases. If you don't have reference codebases, you can set it to `None`.
        """
        # Handle None context_variables
        if context_variables is None:
            context_variables = {}
        if "notes" not in context_variables:
            context_variables["notes"] = []

        # Ensure the last note exists before accessing it
        if len(context_variables["notes"]) == 0:
            # If notes is empty, create a new note with the academic definition
            context_variables["notes"].append({"definition": academic_definition})

        # Ensure all required keys exist in the last note
        if "math_formula" not in context_variables["notes"][-1]:
            context_variables["notes"][-1]["math_formula"] = ""
        if "reference_papers" not in context_variables["notes"][-1]:
            context_variables["notes"][-1]["reference_papers"] = ""

        # context_variables["notes"] = {
        #     academic_definition: {
        #         "definition": academic_definition,
        #         "math_formula": math_formula,
        #         "code_implementation": code_implementation,
        #         "references": references,
        #     }
        # }
        context_variables["notes"][-1]["code_implementation"] = code_implementation
        context_variables["notes"][-1]["reference_codebases"] = reference_codebases
        ret_val = f"""\
I have taken notes for the innovation.
The notes are as follows:
## Academic Definition
{academic_definition}
## Math Formula
{context_variables["notes"][-1]["math_formula"]}
## Reference papers
{context_variables["notes"][-1]["reference_papers"]}
## Code Implementation
{context_variables["notes"][-1]["code_implementation"]}
## Reference codebases
{context_variables["notes"][-1]["reference_codebases"]}
"""
        return Result(
            value=ret_val,
            context_variables=context_variables,
            agent=survey_agent,
        )

    def transfer_to_paper_survey_agent(
        academic_definition: str, context_variables: dict = None
    ):
        """
        Pass a specific academic definition to the `Paper Survey Agent` to extract math formulas.
        [IMPORTANT] Use only after you have actually reviewed the codebases (avoid premature handoff).
        """
        # Handle None context_variables
        if context_variables is None:
            context_variables = {}
        if "notes" not in context_variables:
            context_variables["notes"] = []
        ret_val = f"""\
You should explore the papers and extract the math formula for the academic definition: {academic_definition}.
"""
        context_variables["notes"].append({"definition": academic_definition})
        return Result(
            value=ret_val,
            agent=paper_survey_agent,
            context_variables=context_variables,
        )

    def transfer_to_code_survey_agent(
        academic_definition: str,
        math_formula: str,
        reference_papers: List[str],
        context_variables: dict = None,
    ):
        """
        Pass a specific academic definition and its math formula to the `Code Survey Agent`
        to find the corresponding code implementation.
        [IMPORTANT] You can use this function only after you have use the provided tools to actually and carefully read and analyze the papers. DONNOT use this function before you have read the papers.
        Args:
            academic_definition: the academic definition to be implemented. It should be a single, atomic academic concept with a few words.
            math_formula: the full math formula to be implemented. [IMPORTANT] It should be as complete as possible and it should be the real math formula.
            reference_papers: the list of reference papers. If you don't have reference papers, you can set it to `None`.
        """
        # Handle None context_variables
        if context_variables is None:
            context_variables = {}
        if "notes" not in context_variables:
            context_variables["notes"] = []

        # Ensure the last note exists before accessing it
        if len(context_variables["notes"]) == 0:
            # If notes is empty, create a new note with the academic definition
            context_variables["notes"].append({"definition": academic_definition})

        ret_val = f"""\
You should explore the codebases and extract the code implementation for the academic definition: {academic_definition} and math formula: {math_formula}.
"""
        context_variables["notes"][-1]["math_formula"] = math_formula
        context_variables["notes"][-1]["reference_papers"] = reference_papers
        return Result(
            value=ret_val,
            agent=code_survey_agent,
            context_variables=context_variables,
        )

    # Survey Agent가 보유할 함수 세트 연결
    survey_agent.functions = [transfer_to_paper_survey_agent, case_resolved]
    paper_survey_agent.functions.append(transfer_to_code_survey_agent)
    code_survey_agent.functions.append(transfer_back_to_survey_agent)

    return survey_agent
