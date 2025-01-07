from abc import ABC

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
  content: str
  platform: str


class CleanedSiteDocument(CleanedDocument):
  link: str

  class Config:
    name = "cleaned_sites"
    category = DataCategory.SITES
    use_vector_index = False
