from clearml import PipelineController
from steps import feature_engineering as fe_steps


def feature_engineering(author_full_names):

  controller = PipelineController(name="feature_engineering",
                                  project="prompt2program",
                                  version="1.0.0",
                                  packages=[""],
                                  repo="")

  # Step 1: Query data
  controller.add_function_step(
    name='query_data_warehouse',
    function=fe_steps.query_data_warehouse,
    function_kwargs=dict(author_full_names=author_full_names),
    function_return=['raw_documents']
  )

  # Step 2: Clean documents
  controller.add_function_step(
    name='clean_documents',
    function=fe_steps.clean_documents,
    function_kwargs=dict(raw_documents='${query_data_warehouse.raw_documents}'),
    function_return=['cleaned_documents'],
    parents=["query_data_warehouse"]
  )

  # Step 3: Load to vector DB
  last_step_1 = controller.add_function_step(
    name='load_to_vector_db_1',
    function=fe_steps.load_to_vector_db,
    function_kwargs=dict(documents='${clean_documents.cleaned_documents}'),
    function_return=['invocation_id'],
    parents= ["clean_documents"]
  )

  # Step 4: Chunk and embed
  controller.add_function_step(
    name='chunk_and_embed',
    function=fe_steps.chunk_and_embed,
    function_kwargs=dict(cleaned_documents='${clean_documents.cleaned_documents}'),
    function_return=['embedded_documents'],
    parents=["clean_documents"]
  )

  # Step 5: Load to vector DB
  last_step_2 = controller.add_function_step(
    name='load_to_vector_db_2',
    function=fe_steps.load_to_vector_db,
    function_kwargs=dict(documents='${chunk_and_embed.embedded_documents}'),
    function_return=['upload_successful'],
    parents=["chunk_and_embed"]
  )

  # Start the pipeline
  controller.start_locally(run_pipeline_steps_locally=True)

  return [last_step_1, last_step_2]


if __name__ == "__main__":
  print(f"Feature engineering pipeline started...")
  feature_engineering(["Admin User"])