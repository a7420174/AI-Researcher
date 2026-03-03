from paper_agent.methodology_composing_using_template import methodology_composing
from paper_agent.related_work_composing_using_template import related_work_composing
from paper_agent.experiments_composing import experiments_composing
from paper_agent.introduction_composing import introduction_composing
from paper_agent.conclusion_composing import conclusion_composing
from paper_agent.abstract_composing import abstract_composing
import asyncio
import argparse
import os
from dotenv import load_dotenv
from paper_agent.writing_fix import clean_tex_files_in_folder, process_tex_file
from paper_agent.tex_writer import compile_latex_project

load_dotenv()

PAPER_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PAPER_FILE = os.path.join(
    os.path.dirname(PAPER_AGENT_ROOT), "workplace_paper", "paper.pdf"
)
PAPER_FILE = os.getenv("PAPER_FILE", DEFAULT_PAPER_FILE)


async def writing(
    research_field: str,
    instance_id: str,
    agent_dir: str = None,
    model_dir: str = None,
):
    """
    Compose and generate paper sections.

    Args:
        research_field: The research field (e.g., 'research')
        instance_id: The instance identifier
        agent_dir: Optional agent directory from research (overrides default path)
        model_dir: Optional model directory from research (overrides default path)
    """
    # Use PAPER_FILE env var for output, or default to research_field/instance_id based path
    target_base = (
        os.path.dirname(PAPER_FILE)
        if PAPER_FILE
        else os.path.join(PAPER_AGENT_ROOT, research_field)
    )
    # Include research_field in path to match composing functions
    target_folder = os.path.join(
        target_base, research_field, "target_sections", instance_id
    )

    # Set environment variable for composing functions to find the correct output directory
    os.environ["PAPER_TARGET_FOLDER"] = target_folder
    os.environ["PAPER_RESEARCH_FIELD"] = research_field

    print(f"[DEBUG] target_folder: {target_folder}", flush=True)

    os.makedirs(target_folder, exist_ok=True)

    try:
        await methodology_composing(research_field, instance_id, agent_dir, model_dir)
        await related_work_composing(research_field, instance_id, agent_dir)
        await experiments_composing(research_field, instance_id, agent_dir)
        await introduction_composing(research_field, instance_id)
        await conclusion_composing(research_field, instance_id)
        await abstract_composing(research_field, instance_id)

        clean_tex_files_in_folder(target_folder)

        tex_file_path = os.path.join(target_folder, "related_work.tex")
        bib_file_path = os.path.join(target_folder, "iclr2025_conference.bib")
        if os.path.exists(tex_file_path):
            process_tex_file(tex_file_path, bib_file_path)
        else:
            print(f"Warning: {tex_file_path} not found, skipping tex processing")

        project_directory = target_folder
        main_file = "iclr2025_conference.tex"
        if os.path.exists(os.path.join(project_directory, main_file)):
            compile_latex_project(project_directory, main_file)
        else:
            print(
                f"Warning: {main_file} not found in {project_directory}, skipping latex compilation"
            )
    finally:
        # Clean up environment variables
        os.environ.pop("PAPER_TARGET_FOLDER", None)
        os.environ.pop("PAPER_RESEARCH_FIELD", None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--research_field", type=str, default="research")
    parser.add_argument("--instance_id", type=str, default="default")
    parser.add_argument("--agent_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(
        writing(args.research_field, args.instance_id, args.agent_dir, args.model_dir)
    )
