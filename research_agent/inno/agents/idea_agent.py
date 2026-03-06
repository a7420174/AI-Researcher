from __future__ import annotations

from typing import List

from research_agent.inno.types import Agent, Result

# 환경 주입 래퍼와 환경 타입만 유지
from research_agent.inno.tools.file_surfer_tool import with_env as with_env_file
from research_agent.inno.environment.docker_env import DockerEnv
from research_agent.inno.environment.markdown_browser import RequestsMarkdownBrowser

# 레지스트리 API
from research_agent.inno.registry import (
    register_agent,
    get_tools,
)


# --------------------------------------------------------------------
# 2) Idea Generation Agent  (레지스트리 등록)
#    - 논문 리뷰 기반 혁신 아이디어 생성/선정/고도화
# --------------------------------------------------------------------
@register_agent("get_idea_agent")
def get_idea_agent(model: str, **kwargs):
    file_env: RequestsMarkdownBrowser = kwargs.get("file_env", None)
    assert file_env is not None, "file_env is required"
    web_env = kwargs.get("web_env", None)  # (옵션) 웹 도구 주입 시 사용

    def instructions(context_variables):
        return f"""\
You are an `Idea Generation Agent` specialized in analyzing academic papers located in `{file_env.docker_workplace}/papers/` and generating innovative ideas. Your task is to either:
1. Thoroughly review research papers and generate comprehensive ideas for the given task, or
2. Analyze multiple existing ideas and select/enhance the most novel one.

OBJECTIVE:
For New Idea Generation:
- Conduct thorough literature review of provided papers
- Identify research gaps and challenges
- Generate innovative and feasible ideas
- Provide detailed technical solutions

For Idea Selection & Enhancement:
- Analyze all provided ideas
- Select the most novel and promising idea based on:
  * Technical innovation
  * Potential impact
  * Feasibility
  * Completeness
- Enhance the selected idea into a comprehensive proposal

AVAILABLE TOOLS:
1. Paper Navigation:
   - `open_local_file`: Open and read paper files
   - `page_up_markdown` / `page_down_markdown`: Navigate through pages
   - `find_on_page_ctrl_f` / `find_next`: Search specific content

2. Content Analysis:
   - `question_answer_on_whole_page`: Ask specific questions about the paper

WORKFLOW:
1. Task Identification:
   - If given papers: Proceed with literature review
   - If given multiple ideas: Proceed with idea selection & enhancement

2. For Literature Review:
   - Thoroughly read and analyze all provided papers
   - Extract key concepts, methods, and results
   - Identify research trends and gaps

3. For Idea Selection:
   - Analyze all provided ideas
   - Score each idea on novelty, feasibility, and completeness
   - Select the most promising idea for enhancement

4. Idea Generation/Enhancement:
   Generate/Enhance into a comprehensive proposal including:

   a) Challenges:
   - Current technical limitations
   - Unsolved problems in existing work
   - Key bottlenecks in the field

   b) Existing Methods:
   - Summary of current approaches
   - Their advantages and limitations
   - Key techniques and methodologies used

   c) Motivation:
   - Why the problem is important
   - What gaps need to be addressed
   - Potential impact of the solution

   d) Proposed Method:
   - Detailed technical solution
   - Step-by-step methodology
   - Mathematical formulations (if applicable)
   - Key innovations and improvements
   - Expected advantages over existing methods
   - Implementation considerations
   - Potential challenges and solutions

   e) Technical Details:
   - Architectural design
   - Algorithm specifications
   - Data flow and processing steps
   - Performance optimization strategies

   f) Expected Outcomes:
   - Anticipated improvements
   - Evaluation metrics
   - Potential applications

5. Knowledge Transfer:
   After completing analysis and idea development, use `transfer_to_code_survey_agent` for implementation research.

REQUIREMENTS:
- Be comprehensive in analysis
- Ensure ideas are novel yet feasible
- Provide detailed technical specifications
- Include mathematical formulations when relevant
- Make clear connections between challenges and solutions
- For idea selection: Clearly explain selection criteria and enhancements

Remember: Your output will guide the implementation phase. Be thorough, innovative, and practical in your approach.
"""

    file_tool_names = [
        "open_local_file",
        "page_up_markdown",
        "page_down_markdown",
        "find_on_page_ctrl_f",
        "find_next",
        "question_answer_on_whole_page",
    ]
    tool_list = get_tools(file_tool_names, env=file_env, env_wrapper=with_env_file)

    if web_env is not None:
        web_tool_names = [
            "google_scholar_search",
            "download_from_pdf_link",
        ]
        tool_list.extend(
            get_tools(web_tool_names, env=web_env, env_wrapper=with_env_web)
        )

    return Agent(
        name="Idea Generation Agent",
        model=model,
        instructions=instructions,
        functions=tool_list,
        tool_choice="auto",
        parallel_tool_calls=False,
    )
