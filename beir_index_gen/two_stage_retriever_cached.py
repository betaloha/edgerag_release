from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, Document, QueryBundle
from typing import Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQueryMode,
)

# Similarity score sorting helping function
def getScore(e):
    return e.score

class TwoStageRetrieverCached(VectorIndexRetriever):
    def __init__(self,
        index: VectorStoreIndex,
        similarity_top_k: int = 2,
        similarity_top_k_second_stage: int = 1,
        vector_store_query_mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT,
        filters: Optional[MetadataFilters] = None,
        alpha: Optional[float] = None,
        node_ids: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        sparse_top_k: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        embed_model: Optional[BaseEmbedding] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""       
        # Candidate Nodes cache
        self._indexed_node_cache = dict({0:object()})
        self._similarity_top_k_second_stage = similarity_top_k_second_stage
        # Unique doc id for secondary retrieval
        self._doc_id = 0
        # Stats
        self._cache_hit_count = 0
        self._cache_miss_count = 0
        # Init super
        super().__init__(
            index = index,
            similarity_top_k = similarity_top_k,
            vector_store_query_mode = vector_store_query_mode,
            filters = filters,
            alpha = alpha,
            node_ids = node_ids,
            doc_ids = doc_ids,
            sparse_top_k = sparse_top_k,
            callback_manager = callback_manager,
            object_map = object_map,
            embed_model = embed_model,
            verbose = verbose,
            kwargs = kwargs,
        )

    def _get_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = self._vector_store.query(query, **self._kwargs)
        node_list = self._build_node_list_from_query_result(query_result)
        nodes_with_scores = []
        # Second level query
        for large_node in node_list:
            node_id = large_node.node_id
            cached_index = self._indexed_node_cache.get(node_id)
            # If the cache is missed, create index and add the index to the cache
            if(cached_index == None):
                # Index the subnode if the cache lookup is missed.
                raw_text = large_node.get_text()
                added_doc = Document(text=raw_text)
                # added_doc.metadata["doc_id"] = large_node.metadata["doc_id"]
                id1 = large_node.metadata.get("doc_id", None)
                if(id1 != None):
                    added_doc.metadata["doc_id"] = id1
                small_index = VectorStoreIndex.from_documents([added_doc],
                                                            #   chunk_size = 256,
                                                            #   chunk_overlap = 32,
                                                              similarity_top_k = self._similarity_top_k_second_stage,
                                                              embed_model = self._embed_model,
                                                              )
                self._indexed_node_cache[node_id] = small_index
                self._cache_miss_count = self._cache_miss_count + 1
                cached_index = small_index
                # TODO: Implement replacement policy
            else:
                self._cache_hit_count = self._cache_hit_count + 1
                pass
            ######################################################
            # Catch Error
            if(cached_index==None):
                print("Cache lookup failed, exiting")
                exit()
            
            # Perform secondary query on each index
            # Change k-value of the query
            query.similarity_top_k = self._similarity_top_k_second_stage
            candidate_node_query_result = cached_index.vector_store.query(query)
            candidate_node_query_result.nodes = cached_index.docstore.get_nodes(candidate_node_query_result.ids)
            
            for idx, node in enumerate(candidate_node_query_result.nodes):
                score: Optional[float] = None
                if candidate_node_query_result.similarities is not None:
                    score = candidate_node_query_result.similarities[idx]
            newNode = NodeWithScore(node=node, score=score)
            # newNode.metadata["doc_id"] = node.metadata["doc_id"]
            id1 = node.metadata.get("doc_id", None)
            if(id1 != None):
                newNode.metadata["doc_id"] = id1
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        nodes_with_scores.sort(key=getScore, reverse=True)
        return nodes_with_scores[0:self._similarity_top_k]

    async def _aget_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        print("Not implemented")
        exit()
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = await self._vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)
    
    def get_cache_hit_count():
        return self._cache_hit_count
    
    def get_cache_miss_count():
        return self._cache_miss_count
