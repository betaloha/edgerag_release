from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex, Document, QueryBundle, StorageContext
from typing import Any, List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQueryMode,
)
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore

class TwoStageRetriever(VectorIndexRetriever):
    def __init__(self,
        index: VectorStoreIndex,
        similarity_top_k: int = 1,
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

        # Second level query
        retrieved_documents = []
        for large_node in node_list:
            raw_text = large_node.get_text()
            newDoc = Document(text=raw_text)
            id1 = large_node.metadata.get("doc_id", None)
            if(id1 != None):
                newDoc.metadata["doc_id"] = id1
            retrieved_documents.append(newDoc)
        faiss_index = faiss.IndexFlatL2(len(query_bundle_with_embeddings.embedding))
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        second_index = VectorStoreIndex.from_documents(retrieved_documents, 
                                                    #    chunk_size = 256, 
                                                    #    chunk_overlap = 32, 
                                                       similarity_top_k = self._similarity_top_k_second_stage, 
                                                       embed_model = self._embed_model)
        # Change k-value of the query
        query.similarity_top_k = self._similarity_top_k_second_stage
        second_query_result = second_index.vector_store.query(query)
        second_query_result.nodes = second_index.docstore.get_nodes(second_query_result.ids)
        nodes_with_scores = []
        for idx, node in enumerate(second_query_result.nodes):
            score: Optional[float] = None
            if second_query_result.similarities is not None:
                score = second_query_result.similarities[idx]
            newNode = NodeWithScore(node=node, score=score)
            id1 = node.metadata.get("doc_id", None)
            if(id1 != None):
                newNode.metadata["doc_id"] = id1
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores

    async def _aget_nodes_with_embeddings(
        self, query_bundle_with_embeddings: QueryBundle
    ) -> List[NodeWithScore]:
        print("Not implemented")
        exit()
        query = self._build_vector_store_query(query_bundle_with_embeddings)
        query_result = await self._vector_store.aquery(query, **self._kwargs)
        return self._build_node_list_from_query_result(query_result)
