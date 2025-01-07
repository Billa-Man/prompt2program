from typing import Any, Iterator

from ollama import Client

from application.rag.retriever import ContextRetriever
from domain.embedded_chunks import EmbeddedChunk
from settings import settings


def call_llm_service(query: str, context: str | None) -> Iterator:
  prompt = """
    You are a content creator. Write what the user asked you to while using the provided context as the primary source of information for the content.
    User query: {query}
    Context: {context}
  """

  llm_input = prompt.format(query=query, context=context)

  client = Client(host=settings.OLLAMA_CLIENT_HOST)

  stream = client.chat(model=settings.HUGGINGFACE_MODEL_ID, messages=[
    {
      'role': 'user',
      'content': llm_input,
    },
  ], stream=True)

  return stream


def rag(query: str) -> Any:
  retriever = ContextRetriever(mock=False)
  documents = retriever.search(query, k=5)
  context = EmbeddedChunk.to_context(documents)

  answer_stream = call_llm_service(query, context)

  return answer_stream