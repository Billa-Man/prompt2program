from pydantic import UUID4, Field

from .base import VectorBaseDocument
from .types import DataCategory


class Query(VectorBaseDocument):
  content: str
  metadata: dict = Field(default_factory=dict)

  class Config:
    category = DataCategory.QUERIES

  @classmethod
  def from_str(cls, query: str) -> "Query":
    return Query(content=query.strip("\n "))

  def replace_content(self, new_content: str) -> "Query":
    return Query(
      id=self.id,
      content=new_content,
      metadata=self.metadata,
    )


class EmbeddedQuery(Query):
  embedding: list[float]

  class Config:
    category = DataCategory.QUERIES