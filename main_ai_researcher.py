import numpy as np
import argparse
import os
import asyncio
import json
import re
import sys
import global_state
from dotenv import load_dotenv
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import COMPLETION_MODEL, CHEEP_MODEL, MODULE_DESCRIPTIONS, STOP_WORDS


class InitGuard:
    def __enter__(self):
        # 단일 진입 보장
        with global_state.INIT_LOCK:
            if global_state.INIT_FLAG:
                raise RuntimeError("Another run is in progress.")
            global_state.INIT_FLAG = True
        # chdir 보존
        self._cwd = os.getcwd()
        return self

    def __exit__(self, exc_type, exc, tb):
        # 작업 디렉터리 복구가 항상 먼저
        try:
            os.chdir(self._cwd)
        finally:
            # 플래그 복구도 항상 보장
            with global_state.INIT_LOCK:
                global_state.INIT_FLAG = False
        # 예외 전파(로깅은 호출부에서)
        return False


def init_ai_researcher():
    a = 1


def get_args_research():
    parser = argparse.ArgumentParser()
    parser.add_argument("--container_name", type=str, default="paper_eval")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06")
    parser.add_argument("--workplace_name", type=str, default="workplace")
    parser.add_argument("--cache_path", type=str, default="cache")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--max_iter_times", type=int, default=0)
    parser.add_argument("--use_docker", type=bool, default=True)
    args = parser.parse_args()
    return args


def _extract_keywords_from_topic(topic: str) -> list:
    """
    Extract search keywords from a topic string using LLM.
    Identifies the most important research-related terms for paper search.
    """
    import logging
    import sys
    import os

    try:
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        sub_dir = os.path.join(current_dir, "research_agent")
        sys.path.insert(0, sub_dir)

        from research_agent.constant import CHEEP_MODEL
        from openai import OpenAI

        client = OpenAI()

        prompt = f"""Extract the most important research keywords from the following topic for academic paper search.

Topic: {topic}

Your task:
1. Identify 3-5 key research concepts or terms that are most relevant for finding academic papers
2. Include important technical terms, methods, algorithms, or domain-specific concepts
3. Exclude common words like "research", "study", "analysis", "paper", "method", "approach"
4. Return ONLY the keywords, one per line, no numbering or bullets

Example output for "How to improve image classification using transformer attention":
transformer
attention mechanism
image classification
Vision Transformer
convolutional neural network"""

        response = client.chat.completions.create(
            model=CHEEP_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        keywords_text = response.choices[0].message.content.strip()
        keywords = [line.strip() for line in keywords_text.split("\n") if line.strip()]

        logging.info(f"LLM extracted keywords: {keywords}")
        return keywords

    except Exception as e:
        logging.warning(
            f"LLM keyword extraction failed: {e}. Using fallback regex method."
        )

        # Fallback to regex-based extraction
        topic_cleaned = re.sub(r"[^\w\s-]", " ", topic)
        words = topic_cleaned.split()

        meaningful_words = [
            word
            for word in words
            if len(word.lower().strip("-")) >= 3
            and word.lower().strip("-") not in STOP_WORDS
            and not word.isdigit()
        ]

        if len(meaningful_words) >= 2:
            two_word_phrases = [
                f"{meaningful_words[i]} {meaningful_words[i + 1]}"
                for i in range(len(meaningful_words) - 1)
            ]
            return two_word_phrases + meaningful_words

        return meaningful_words[:5] if meaningful_words else [topic]


def _find_references_for_topic(topic: str) -> str:
    """
    Find relevant reference papers for a given topic using available search tools.
    Uses BioMCP and OpenAlex to find papers.
    Returns a string of found references in a format suitable for the research agents.
    """
    import logging

    references = []

    try:
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        sub_dir = os.path.join(current_dir, "research_agent")
        os.chdir(sub_dir)

        from research_agent.inno.registry import get_tool

        keywords = _extract_keywords_from_topic(topic)

        genes = [
            kw for kw in keywords if kw.isupper() and len(kw) >= 2 and len(kw) <= 10
        ]
        search_keywords = [kw for kw in keywords if kw not in genes]

        # Try OpenAlex first for academic papers
        try:
            openalex_search_papers = get_tool("openalex_search_papers")
            for keyword in search_keywords[:5]:
                try:
                    result = openalex_search_papers(query=keyword, limit=5)
                    if result and "Error" not in str(result):
                        references.append(f"OpenAlex Search for '{keyword}':\n{result}")
                except Exception as e:
                    logging.warning(f"OpenAlex search failed for '{keyword}': {e}")
        except Exception as e:
            logging.warning(f"OpenAlex tools not available: {e}")

        # Try BioMCP for biomedical papers
        try:
            biomcp_article_search = get_tool("biomcp_article_search")
            for gene in genes[:3]:
                try:
                    result = biomcp_article_search(gene=gene, limit=5)
                    if result and "Error" not in result:
                        references.append(f"BioMCP Gene Search for {gene}:\n{result}")
                except Exception as e:
                    logging.warning(f"BioMCP search failed for gene '{gene}': {e}")
        except Exception as e:
            logging.warning(f"BioMCP tools not available: {e}")

        # Try arxiv as fallback
        try:
            from research_agent.inno.tools.arxiv import search_arxiv

            for keyword in search_keywords[:5]:
                try:
                    arxiv_results = search_arxiv(keyword, max_results=5)
                    if arxiv_results and isinstance(arxiv_results, list):
                        for item in arxiv_results[:3]:
                            if isinstance(item, dict):
                                title = item.get("title", "")
                                arxiv_id = item.get("id", "")
                                if title and arxiv_id:
                                    ref = f"{title} (arXiv: {arxiv_id})"
                                    references.append(ref)
                except Exception as e:
                    logging.warning(f"ArXiv search failed for '{keyword}': {e}")
        except Exception as e:
            logging.warning(f"ArXiv tools not available: {e}")

        # If no results, try full topic search with OpenAlex
        if not references:
            try:
                openalex_search_papers = get_tool("openalex_search_papers")
                result = openalex_search_papers(query=topic, limit=10)
                if result and "Error" not in str(result):
                    references.append(f"OpenAlex Search for '{topic}':\n{result}")
            except Exception as e:
                logging.warning(f"OpenAlex topic search failed: {e}")

    except Exception as e:
        logging.warning(f"Error finding references: {e}")

    if not references:
        return ""

    return "\n".join(references)


def main_ai_researcher(
    input, reference, mode, research_field="general", use_docker=None
):
    load_dotenv()
    container_name = os.getenv("CONTAINER_NAME")
    workplace_name = os.getenv("WORKPLACE_NAME")
    cache_path = os.getenv("CACHE_PATH")
    port = int(os.getenv("PORT"))
    max_iter_times = int(os.getenv("MAX_ITER_TIMES"))
    if use_docker is None:
        use_docker = os.getenv("USE_DOCKER", "true").lower() == "true"

    match mode:
        case "Detailed Idea Description":
            found_reference = reference
            if not reference or not reference.strip():
                import logging

                logging.warning(
                    "No reference papers provided. Attempting to find relevant references for the topic..."
                )
                found_reference = _find_references_for_topic(input)
                if found_reference:
                    logging.info(
                        f"Found {len(found_reference.split(chr(10)))} relevant references for the topic."
                    )
                else:
                    logging.warning(
                        "Could not find relevant references. Proceeding without references."
                    )

            with InitGuard():
                global_state.INIT_FLAG = True
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent import run_infer_plan

                args = get_args_research()
                args.model = COMPLETION_MODEL
                args.container_name = container_name
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.use_docker = use_docker

                project_info = run_infer_plan.main(
                    args, input, found_reference or reference, input
                )

                # After research completes, run writing module
                from paper_agent import writing

                instance_id = project_info.get("instance_id", "query_based")
                asyncio.run(
                    writing.writing(
                        research_field,
                        instance_id,
                        agent_dir=project_info.get("agent_dir"),
                        model_dir=project_info.get("model_dir"),
                    )
                )

                return "Research and paper writing completed successfully"

        case "Reference-Based Ideation":
            found_reference = reference
            if not reference or not reference.strip():
                import logging

                logging.warning(
                    "No reference papers provided. Attempting to find relevant references for the topic..."
                )
                found_reference = _find_references_for_topic(input)
                if found_reference:
                    logging.info(
                        f"Found {len(found_reference.split(chr(10)))} relevant references for the topic."
                    )
                else:
                    logging.warning(
                        "Could not find relevant references. Proceeding without references."
                    )

            with InitGuard():
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent import run_infer_idea

                args = get_args_research()

                args.container_name = container_name
                args.model = COMPLETION_MODEL
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.use_docker = use_docker

                project_info = run_infer_idea.main(
                    args, found_reference or reference, input
                )

                # After research completes, run writing module
                from paper_agent import writing

                instance_id = project_info.get("instance_id", "query_based")
                asyncio.run(
                    writing.writing(
                        research_field,
                        instance_id,
                        local_root=project_info.get("local_root"),
                        agent_dir=project_info.get("agent_dir"),
                        model_dir=project_info.get("model_dir"),
                    )
                )

                return "Research and paper writing completed successfully"

        case "Deep Research":
            with InitGuard():
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent import run_deep_research
                from constant import COMPLETION_MODEL

                result_info = run_deep_research.main(topic=input, reference=reference)

                # Handle both old string return and new dict return
                if isinstance(result_info, dict):
                    result = result_info.get("result", "")

                    # Create agent JSON file for paper writing (Deep Research mode)
                    agent_json_path = os.path.join(result_info.get("agent_dir", ""), "deep_research_0.json")
                    agent_data = {
                        "messages": [
                            {"role": "user", "content": input},
                            {"role": "assistant", "content": result}
                        ],
                        "context_variables": {
                            "topic": input,
                            "research_result": result
                        }
                    }
                    with open(agent_json_path, "w", encoding="utf-8") as f:
                        json.dump(agent_data, f, ensure_ascii=False, indent=2)

                    # Change back to project root before running paper writing
                    os.chdir(current_dir)

                    # Run paper writing with proper paths
                    from paper_agent import writing

                    asyncio.run(
                        writing.writing(
                            research_field,
                            instance_id,
                            local_root=result_info.get("local_root", ""),
                            agent_dir=result_info.get("agent_dir", ""),
                            model_dir=result_info.get("model_dir", ""),
                        )
                    )

                    return f"Deep research and paper writing completed successfully."
                return result_info


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Research topic or input"
    )
    parser.add_argument("--reference", type=str, default="", help="Reference papers")
    parser.add_argument(
        "--mode",
        type=str,
        default="Detailed Idea Description",
        choices=[
            "Detailed Idea Description",
            "Reference-Based Ideation",
            "Deep Research",
        ],
        help="Research mode",
    )
    parser.add_argument(
        "--research_field",
        type=str,
        default="general",
        help="Research field for paper writing (default: general)",
    )
    args = parser.parse_args()

    result = main_ai_researcher(
        args.input, args.reference, args.mode, args.research_field
    )
    print(f"Result: {result}")
