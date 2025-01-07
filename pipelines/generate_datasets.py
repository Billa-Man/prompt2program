from clearml import PipelineController

from settings import settings
from steps import generate_datasets as cd_steps


def generate_datasets(
  test_split_size: float = 0.1,
  push_to_huggingface: bool = False,
  dataset_id: str | None = None,
) -> None:
  pipeline = PipelineController(name="generate_datasets",
                                project="prompt2program",
                                version="1.0.0",
                                packages=[""],
                                repo="")

  pipeline.add_function_step(
    name='query_feature_store',
    function=cd_steps.query_feature_store,
    function_return=['cleaned_documents']
  )

  pipeline.add_function_step(
    name='create_prompts',
    function=cd_steps.create_prompts,
    function_kwargs=dict(documents='${query_feature_store.cleaned_documents}'),
    function_return=['prompts'],
    parents=["query_feature_store"]
  )

  pipeline.add_function_step(
    name='generate_instruction_dataset',
    function=cd_steps.generate_instruction_dataset,
    function_kwargs=dict(prompts='${create_prompts.prompts}', test_split_size=test_split_size),
    function_return=['dataset'],
    parents=["create_prompts"]
  )

  if push_to_huggingface:
    pipeline.add_function_step(
      name='push_to_huggingface',
      function=cd_steps.push_to_huggingface,
      function_kwargs=dict(dataset='${generate_instruction_dataset.dataset}', dataset_id=dataset_id),
      parents=["generate_instruction_dataset"]
    )

  pipeline.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
  print(f"Dataset generation pipeline started...")
  generate_datasets(push_to_huggingface=True, dataset_id=settings.DATASET_ID)