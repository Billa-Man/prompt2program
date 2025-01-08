import concurrent.futures

from loguru import logger
from qdrant_client.models import FieldCondition, Filter, MatchValue

from application.preprocessing.dispatchers import EmbeddingDispatcher
from domain.embedded_chunks import (
  EmbeddedChunk,
  EmbeddedSiteChunk,
)
from domain.queries import EmbeddedQuery, Query
from .query_expansion import QueryExpansion
from .reranking import Reranker
from .. import utils


class ContextRetriever:
  def __init__(self, mock: bool = False) -> None:
    self._query_expander = QueryExpansion(mock=mock)
    self._reranker = Reranker(mock=mock)

  def search(
    self,
    query: str,
    k: int = 3,
    expand_to_n_queries: int = 3,
  ) -> list:
    query_model = Query.from_str(query)

    query_model = self._metadata_extractor.generate(query_model)

    n_generated_queries = self._query_expander.generate(query_model, expand_to_n=expand_to_n_queries)
    logger.info(
      f"Successfully generated {len(n_generated_queries)} search queries.",
    )

    with concurrent.futures.ThreadPoolExecutor() as executor:
      search_tasks = [executor.submit(self._search, _query_model, k) for _query_model in n_generated_queries]

      n_k_documents = [task.result() for task in concurrent.futures.as_completed(search_tasks)]
      n_k_documents = utils.misc.flatten(n_k_documents)
      n_k_documents = list(set(n_k_documents))

    logger.info(f"{len(n_k_documents)} documents retrieved successfully")

    if len(n_k_documents) > 0:
      k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
    else:
      k_documents = []

    return k_documents

  def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
    assert k >= 3, "k should be >= 3"

    def _search_data_category(
      data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
    ) -> list[EmbeddedChunk]:
      if embedded_query.author_id:
        query_filter = Filter(
          must=[
            FieldCondition(
              key="author_id",
              match=MatchValue(
                value=str(embedded_query.author_id),
              ),
            )
          ]
        )
      else:
        query_filter = None

      return data_category_odm.search(
        query_vector=embedded_query.embedding,
        limit=k // 3,
        query_filter=query_filter,
      )

    embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

    site_chunks = _search_data_category(EmbeddedSiteChunk, embedded_query)

    return site_chunks

  def rerank(self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int) -> list[EmbeddedChunk]:
    if isinstance(query, str):
      query = Query.from_str(query)

    reranked_documents = self._reranker.generate(query=query, chunks=chunks, keep_top_k=keep_top_k)

    logger.info(f"{len(reranked_documents)} documents reranked successfully.")

    return reranked_documents