from abc import ABC, abstractmethod

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

import domain
from application import utils
from domain.cleaned_documents import CleanedDocument
from domain.dataset import TrainTestSplit
from domain.prompt import GenerateDatasetSamplesPrompt, Prompt
from domain.types import DataCategory
from settings import settings
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser


class DatasetGenerator(ABC):
  tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)

  system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
  prompt_template_str: str | None = None

  @classmethod
  def get_system_prompt(cls) -> Prompt:

    dataset_format = (
      "instruction-answer pairs"
    )
    input_variables = {
      "dataset_format": dataset_format,
    }
    system_prompt = cls.system_prompt_template.format(**input_variables)

    return Prompt(
      template=cls.system_prompt_template,
      input_variables=input_variables,
      content=system_prompt,
    )

  @classmethod
  def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
    documents = generation_utils.extract_substrings(documents)

    grouped_prompts = {}
    grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
    for category, category_documents in grouped_cleaned_documents.items():
      category_prompts = [cls.get_prompt(document) for document in category_documents]
      grouped_prompts[category] = category_prompts

    return grouped_prompts

  @classmethod
  def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
    assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

    data_category = document.get_category()

    prompt_template = PromptTemplate.from_template(
      template=cls.prompt_template_str,
      template_format="jinja2",
    )
    input_variables = {
      "extract": document.content,
    }
    prompt = prompt_template.format(**input_variables)
    prompt_tokens = cls.tokenizer.encode(prompt)
    if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
      prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
      prompt = cls.tokenizer.decode(prompt_tokens)

    prompt = GenerateDatasetSamplesPrompt(
      template=prompt_template.template,
      input_variables=input_variables,
      content=prompt,
      num_tokens=len(prompt_tokens),
      data_category=data_category,
      document=document,
    )

    return prompt

  @classmethod
  def generate(
    cls,
    prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
    test_size: float = 0.2
  ) -> TrainTestSplit:

    def _to_langchain(
      prompt: GenerateDatasetSamplesPrompt,
    ) -> list[BaseMessage]:
      messages = [
        SystemMessage(content=cls.get_system_prompt().content),
        HumanMessage(content=prompt.content),
      ]

      return messages

    assert settings.OPENAI_API_KEY is not None, "OpenAI API key must be set to generate datasets"

    llm = ChatOpenAI(
      model=settings.OPENAI_MODEL_ID,
      api_key=settings.OPENAI_API_KEY,
      max_tokens=1200,
      temperature=0.7,
    )
    parser = ListPydanticOutputParser(pydantic_object=cls._get_dataset_sample_type())

    chain = llm | parser

    datasets = {}
    for category, category_prompts in prompts.items():
      langchain_category_prompts = [_to_langchain(prompt) for prompt in category_prompts]
      batches = utils.misc.batch(langchain_category_prompts, size=24)

      flattened_instruct_dataset_samples = []
      for batch in batches:
        try:
          batched_dataset_samples = chain.batch(batch, stop=None)

          for instruct_dataset_sample_batch in batched_dataset_samples:
            flattened_instruct_dataset_samples.extend(instruct_dataset_sample_batch)
        except OutputParserException:
          logger.exception(f"Failed to parse the output JSON for a batch for category {category}")

      dataset = domain.dataset.build_dataset(
        category=category, samples=flattened_instruct_dataset_samples
      )
      datasets[category] = dataset
      logger.info(f"Generated {len(dataset.samples)} samples for category '{category}'.")

    processed_datasets = cls.post_process_datasets(datasets, test_size=test_size)

    return processed_datasets

  @classmethod
  def _get_dataset_sample_type(
    cls,
  ) -> type[domain.dataset.InstructDatasetSample]:
    return (
      domain.dataset.InstructDatasetSample
    )

  @classmethod
  @abstractmethod
  def post_process_datasets(
    cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
  ) -> TrainTestSplit:
    pass


class InstructionDatasetGenerator(DatasetGenerator):
  prompt_template_str = """Based on the following extracts, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
    
Example instruction: Explain the concept of a ROS2 Twin. \
Example answer: A ROS2 Twin is essentially a digital replica of a robotic system's behavior, state, and interactions. \
It mimics the real robot's operations within a simulated environment, allowing developers to test, monitor, and optimize without physical hardware. \
The idea is to create a virtual counterpart of the robot using ROS2 tools, like Gazebo for simulation and Nav2 for navigation, to mirror its real-world actions. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{extract}
"""

  @classmethod
  def post_process_datasets(
    cls, datasets: dict[DataCategory, domain.dataset.InstructDataset], test_size: float
  ) -> TrainTestSplit:
    train_test_split = generation_utils.create_instruct_train_test_split(
      datasets, test_size=test_size, random_state=42
    )

    return train_test_split


def get_dataset_generator() -> type[DatasetGenerator]:
  return InstructionDatasetGenerator