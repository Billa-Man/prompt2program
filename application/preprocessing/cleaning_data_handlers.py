from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from domain.cleaned_documents import (
  CleanedDocument,
  CleanedSiteDocument,
)
from domain.documents import (
  Document,
  SiteDocument,
)
from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
  """
  Abstract class for all cleaning data handlers.
  All data transformations logic for the cleaning step is done here
  """

  @abstractmethod
  def clean(self, data_model: DocumentT) -> CleanedDocumentT:
    pass


class SiteCleaningHandler(CleaningDataHandler):
  def clean(self, data_model: SiteDocument) -> CleanedSiteDocument:
    return CleanedSiteDocument(
      id=data_model.id,
      content=clean_text(" #### ".join(data_model.content.values())),
      platform=data_model.platform,
      link=data_model.link,
    )