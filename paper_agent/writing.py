from paper_agent.methodology_composing_using_template import methodology_composing
from paper_agent.related_work_composing_using_template import related_work_composing
from paper_agent.experiments_composing import experiments_composing
from paper_agent.introduction_composing import introduction_composing
from paper_agent.conclusion_composing import conclusion_composing
from paper_agent.abstract_composing import abstract_composing
import asyncio
import argparse
from paper_agent.writing_fix import clean_tex_files_in_folder, process_tex_file
from paper_agent.tex_writer import compile_latex_project


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
    await methodology_composing(research_field, instance_id, agent_dir, model_dir)
    await related_work_composing(research_field, instance_id, agent_dir)
    await experiments_composing(research_field, instance_id, agent_dir)
    await introduction_composing(research_field, instance_id)
    await conclusion_composing(research_field, instance_id)
    await abstract_composing(research_field, instance_id)

    target_folder = f"{research_field}/target_sections/{instance_id}"
    clean_tex_files_in_folder(target_folder)

    tex_file_path = f"{research_field}/target_sections/{instance_id}/related_work.tex"
    bib_file_path = (
        f"{research_field}/target_sections/{instance_id}/iclr2025_conference.bib"
    )
    process_tex_file(tex_file_path, bib_file_path)

    project_directory = f"{research_field}/target_sections/{instance_id}"
    main_file = "iclr2025_conference.tex"
    compile_latex_project(project_directory, main_file)


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
