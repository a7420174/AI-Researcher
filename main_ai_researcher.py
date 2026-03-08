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
from constant import COMPLETION_MODEL, CHEEP_MODEL, MODULE_DESCRIPTIONS


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


def _find_references_for_topic(topic: str) -> str:
    """
    Find relevant methodology papers for a given topic using the Reference Finder Agent.
    Returns paper titles found by the agent.
    """
    import logging
    import asyncio

    try:
        current_file_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(current_file_path)
        sub_dir = os.path.join(current_dir, "research_agent")
        os.chdir(sub_dir)
        sys.path.insert(0, sub_dir)

        from research_agent.constant import CHEEP_MODEL
        from research_agent.inno.agents import bootstrap_import
        from research_agent.inno.registry import get_agent_factory
        from research_agent.inno.workflow.flowcache import AgentModule
        from research_agent.inno import MetaChain
        import app_bootstrap

        app_bootstrap.bootstrap_registry()
        bootstrap_import(modules=["research_agent.inno.agents.reference_finder_agent"])

        get_reference_finder_agent_factory = get_agent_factory(
            "get_reference_finder_agent"
        )
        reference_finder_agent = get_reference_finder_agent_factory(model=CHEEP_MODEL)

        client = MetaChain()

        messages = [
            {
                "role": "user",
                "content": f"Find methodology papers for the following topic: {topic}\n\nReturn ONLY the paper titles, one per line, no numbering or bullets.",
            }
        ]

        import tempfile

        cache_dir = tempfile.gettempdir()

        async def run_agent():
            agent_module = AgentModule(
                agent=reference_finder_agent,
                client=client,
                cache_path=cache_dir,
            )
            response = await agent_module(messages=messages, context_variables={})
            return response

        response = asyncio.run(run_agent())

        if isinstance(response, tuple):
            response_messages, response_context = response
            if response_messages and len(response_messages) > 0:
                last_message = response_messages[-1]
                content = last_message.get("content", "")
                return content.strip()
        elif hasattr(response, "messages") and response.messages:
            last_message = response.messages[-1]
            content = last_message.get("content", "")
            return content.strip()

        return ""

    except Exception as e:
        logging.warning(f"Error finding references: {e}")
        import traceback

        traceback.print_exc()
        return ""


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

                # Prevent argparse from parsing sys.argv during import
                import sys

                original_argv = sys.argv.copy()
                sys.argv = [""]

                from research_agent import run_infer_plan
                from research_agent import run_infer_plan as rp_module
                from constant import COMPLETION_MODEL

                # Restore argv after import
                sys.argv = original_argv

                # Create args object directly without parsing sys.argv
                class Args:
                    pass

                args = Args()
                args.model = COMPLETION_MODEL
                args.container_name = container_name
                args.workplace_name = workplace_name
                args.cache_path = cache_path
                args.port = port
                args.max_iter_times = max_iter_times
                args.use_docker = use_docker
                args.conda_path = None
                args.use_conda = False
                args.uv_path = None
                args.venv_path = None

                project_info = rp_module.main(
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

                instance_id = project_info.get("instance_id", "reference_based")
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

                result_info = run_deep_research.main(
                    topic=input, max_iter_times=max_iter_times
                )

                # Handle both old string return and new dict return
                if isinstance(result_info, dict):
                    result = result_info.get("result", "")

                    # Create agent JSON file for paper writing (Deep Research mode)
                    agent_json_path = os.path.join(
                        result_info.get("agent_dir", ""), "deep_research_0.json"
                    )
                    agent_data = {
                        "messages": [
                            {"role": "user", "content": input},
                            {"role": "assistant", "content": result},
                        ],
                        "context_variables": {
                            "topic": input,
                            "research_result": result,
                        },
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
                            instance_id=result_info.get("instance_id", ""),
                            local_root=result_info.get("local_root", ""),
                            agent_dir=result_info.get("agent_dir", ""),
                            model_dir=result_info.get("model_dir", ""),
                        )
                    )

                    paper_target_folder = os.path.join(
                        result_info.get("local_root", ""),
                        research_field,
                        "target_sections",
                        result_info.get("instance_id", ""),
                    )
                    return f"Deep research and paper writing completed successfully.\nPaper output: {paper_target_folder}"
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
