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


def get_args_paper():
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, default="research")
    args = parser.parse_args()
    return args


def _extract_keywords_from_topic(topic: str) -> list:
    """
    Automatically extract search keywords from a topic string.
    Removes common words and extracts meaningful research-related terms.
    """
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
        from research_agent.inno.tools.arxiv import search_arxiv

        keywords = _extract_keywords_from_topic(topic)

        genes = [
            kw for kw in keywords if kw.isupper() and len(kw) >= 2 and len(kw) <= 10
        ]
        search_keywords = [kw for kw in keywords if kw not in genes]

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

        if not search_keywords and not genes:
            try:
                arxiv_results = search_arxiv(topic, max_results=10)
                if arxiv_results and isinstance(arxiv_results, list):
                    for item in arxiv_results[:5]:
                        if isinstance(item, dict):
                            title = item.get("title", "")
                            arxiv_id = item.get("id", "")
                            if title and arxiv_id:
                                ref = f"{title} (arXiv: {arxiv_id})"
                                references.append(ref)
            except Exception as e:
                logging.warning(f"ArXiv search failed for topic '{topic}': {e}")

    except Exception as e:
        logging.warning(f"Error finding references: {e}")

    if not references:
        return ""

    return "\n".join(references)


def main_ai_researcher(input, reference, mode, use_docker=None):
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

                from research_agent import run_infer_idea, run_infer_plan

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

                research_field = "general"
                asyncio.run(
                    writing.writing(
                        research_field,
                        project_info.get("instance_id", "query_based"),
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

                from research_agent import run_infer_idea, run_infer_plan

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

                research_field = "general"
                asyncio.run(
                    writing.writing(
                        research_field,
                        project_info.get("instance_id", "query_based"),
                        agent_dir=project_info.get("agent_dir"),
                        model_dir=project_info.get("model_dir"),
                    )
                )

                return "Research and paper writing completed successfully"

        case "Paper Generation Agent":
            with InitGuard():
                from paper_agent import writing

                args = get_args_paper()

                research_field = "general"
                args.research_field = research_field

                asyncio.run(writing.writing(args.research_field, research_field))

                return "Paper generation completed successfully"

        case "Deep Research":
            with InitGuard():
                current_file_path = os.path.realpath(__file__)
                current_dir = os.path.dirname(current_file_path)
                sub_dir = os.path.join(current_dir, "research_agent")
                os.chdir(sub_dir)

                from research_agent import run_deep_research

                result_info = run_deep_research.main(topic=input, reference=reference)
                # Handle both old string return and new dict return
                if isinstance(result_info, dict):
                    result = result_info.get("result", "")
                    # Optionally run writing for deep research (may not have full project)
                    # For now, just return the research result
                    return result
                return result_info
