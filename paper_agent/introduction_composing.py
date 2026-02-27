import os
import json
import asyncio
import logging
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_agent.gpt_client import GPTClient
from paper_agent.section_composer import SectionComposer, setup_logging


class IntroductionComposer(SectionComposer):
    def __init__(self, research_field: str, structure_iterations: int = 3):
        super().__init__(research_field, "introduction", structure_iterations)

    def read_section_content(self, target_paper: str, section_name: str) -> str:
        """Read content from an existing section file"""
        normalized_title = self.normalize_title(target_paper)
        section_path = f"{self.research_field}/target_sections/{normalized_title}/{section_name}.tex"
        try:
            with open(section_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            logging.warning(f"Section file {section_path} not found")
            return ""

    async def generate_or_revise_structure(self, content, current_structure, iteration):
        prompt = f"""Based on the given content, generate or revise the introduction structure, using latex format.
Current iteration: {iteration}/{self.structure_iterations}

Current structure (if exists):
{current_structure}

Content to analyze:
{content}

Guidelines for structure generation:
1. STRUCTURE FORMAT:
   \section{{Introduction}}
   
   % [Research Background]
   % - Problem statement
   % - Existing methods and their approaches
   
   % [Research Motivation]
   % - Limitations of existing methods
   % - Important challenges to address
   % - Key research questions
   % - Technical opportunities or insights
   
   % [Methodology Overview]
   % - High-level description of proposed approach
   % - How it addresses the identified challenges
   % - Connection to research motivation
   
   % [Contributions]
   % - Key technical innovations
   % - Novel problem formulation
   % - Important findings
   % - Experimental validation

2. CONTENT REQUIREMENTS:
   - Keep structure at a high level without technical details
   - Focus on logical flow from problem to solution
   - Ensure clear connections between different parts
   - Highlight novel aspects and contributions

Output only the LaTeX structure with comments as specified above."""

        return await self.gpt_client.chat(prompt=prompt)

    async def detailize_subsection(
        self, structure, current_text, content, subsection=None
    ):
        writing_template = self.get_random_template()
        prompt = f"""Write a comprehensive introduction section based on the provided structure and content.

CURRENT INTRODUCTION VERSION (if any):
{current_text}

STRUCTURE:
{structure}

NEW CONTENT TO INCORPORATE:
{content}

WRITING TEMPLATE:
{writing_template}

Writing Guidelines:

1. RESEARCH BACKGROUND (2 paragraphs):
   - Begin with broad context of the research area
   - Clearly define the studied problem
   - Present existing methods objectively, categorized in 1-3 research lines. Clearly state the relationship between different research lines
   - Use citations extensively (\cite{{}})

2. RESEARCH MOTIVATION (1-2 paragraphs):
   - Identify clear gaps in existing work (you may use some concrete examples)
   - Present logical progression to your approach
   - Highlight key challenges and research questions
   - Make compelling case for your method

3. METHODOLOGY OVERVIEW (1-2 paragraphs):
   - Introduce proposed method using natural language only
   - Avoid super technical details like hyperparameters and model configurations
   - Connect to identified challenges
   - Use clear, accessible language
   - This part should be long enough

4. CONTRIBUTIONS:
   - Use \begin{{itemize}} environment
   - List 3-4 concrete contributions
   - Be specific but not technical
   - Highlight novelty and significance
   - Include both methodological and empirical contributions

STYLE REQUIREMENTS:
- Maintain formal academic tone
- Write concisely and clearly
- Use present tense for established facts
- Use proper transitions between sections
- Avoid self-references or promotional language
- Follow the instruction to assign enough paragraphs to each aspect

Output the complete LaTeX text for the introduction section. Do not output any other content."""

        return await self.gpt_client.chat(prompt=prompt)

    async def final_writing_checklist(self, introduction_text: str) -> str:
        prompt = f"""Review and revise the introduction section following these guidelines:

Current introduction text:
{introduction_text}

CHECKLIST FOR REVISION:

1. WRITING STYLE:
   - Ensure formal academic tone
   - Remove redundant phrases
   - Strengthen logical flow
   - Check paragraph transitions
   - Verify citation usage

2. CONTENT VERIFICATION:
   - Confirm problem statement clarity
   - Verify motivation strength
   - Check methodology overview clarity
   - Validate contribution claims
   - Ensure balanced coverage

3. STRUCTURAL ELEMENTS:
   - Verify proper LaTeX formatting
   - Check itemize environment
   - Confirm citation commands
   - Review paragraph organization
   - Validate section flow

4. ACADEMIC STANDARDS:
   - Remove promotional language
   - Ensure objective presentation
   - Check technical accuracy
   - Verify scientific rigor
   - Maintain professional tone

5. DO NOT INCLUDE FORMULAS

Output the revised introduction section incorporating all these improvements. Reply with LaTeX text only."""

        return await self.gpt_client.chat(prompt=prompt)

    async def compose_section(self, target_paper: str) -> str:
        # Read existing sections
        methodology = self.read_section_content(target_paper, "methodology")
        related_work = self.read_section_content(target_paper, "related_work")
        experiments = self.read_section_content(target_paper, "experiments")
        content_bundle = methodology + "\n\n" + experiments + "\n\n" + related_work

        # Step 1: Iterative structure generation
        structure = ""

        for iteration in range(self.structure_iterations):
            logging.info(
                f"Structure iteration {iteration + 1}/{self.structure_iterations}"
            )

            structure = await self.generate_or_revise_structure(
                content_bundle, structure, iteration + 1
            )

            self.write_temp_log(structure, f"iteration_{iteration + 1}_final")

        # Step 2: Write complete introduction
        introduction = ""
        introduction = await self.detailize_subsection(
            structure, content_bundle, introduction
        )

        self.write_temp_log(introduction, "initial_introduction")

        # Step 3: Final writing checklist
        final_introduction = await self.final_writing_checklist(introduction)
        self.write_temp_log(final_introduction, "final_introduction")

        # Save final output
        output_dir = f"{self.research_field}/target_sections/{self.normalize_title(target_paper)}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "introduction.tex")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_introduction)
        logging.info(f"Saved final introduction to {output_path}")

        return final_introduction


async def introduction_composing(research_field: str, instance_id: str):
    setup_logging(research_field)

    composer = IntroductionComposer(
        research_field=research_field, structure_iterations=1
    )

    try:
        introduction = await composer.compose_section(instance_id)
        logging.info("Introduction composition completed")
    except Exception as e:
        logging.error(f"Error during introduction composition: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(introduction_composing())
