from abc import ABC
from pydantic import UUID4, Field

from domain.types import DataCategory
from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
  content: str
  embedding: list[float] | None
  platform: str
  document_id: UUID4
  metadata: dict = Field(default_factory=dict)

  @classmethod
  def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
    context = ""
    for i, chunk in enumerate(chunks):
      context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Content: {chunk.content}\n
            """

    return context


class EmbeddedSiteChunk(EmbeddedChunk):
  link: str

  class Config:
    name = "embedded_sites"
    category = DataCategory.SITES
    use_vector_index = True