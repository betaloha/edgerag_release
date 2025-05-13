
from llama_index.core import Document, Settings
from typing import Any, List, Optional, Dict
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
import faiss
import torch
import os
import numpy as np
# Custom retriever compatible with BEIR, not a drop-in replacement for VectorIndexRetriever
class TwoStageRetrieverOffline():
    def __init__(self,
        faiss_index: callable,
        corpus_list: List,
        redir_table: Dict,
        index_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        cost_table: Optional[Dict] = {}, # Does nothing
    ) -> None:
        self.faiss_index = faiss_index
        self.corpus_list = corpus_list
        self.embed_model = embed_model
        self.index_dir = index_dir
        self.redir_table = redir_table
        # Init second-level indexes
        self.second_level_faiss_indexes = {-1:None}
        for i in self.redir_table:
            second_level_index_path = os.path.join(self.index_dir, "second_stage_"+str(i))
            second_level_faiss_index = faiss.read_index(second_level_index_path)
            self.second_level_faiss_indexes[i] =second_level_faiss_index

    def retrieve(
        self,
        query: str,
        top_k: int,
        nprobe: int,

    ) -> List[NodeWithScore]:
        embed_model = None
        if(self.embed_model==None):
            embed_model = Settings.embed_model
        else:
            embed_model = self.embed_model
        # Throw error if there is no embedding.
        if(embed_model==None):
            print("No embedding")
            exit()
        # Generate query embedding
        query_embedding = embed_model.get_text_embedding(query)
        # Convert to fp16
        # query_embedding = np.array([query_embedding], dtype=np.float16)
        query_embedding = torch.tensor([query_embedding])
        # First stage look up
        Dis, Idx = self.faiss_index.search(query_embedding, nprobe)

        # Get doc index from redir table
        doc_ids = []
        embeddings_list = []
        for i in Idx[0]:
            doc_ids += self.redir_table[i]
            # # Retrieve Second-level index
            # second_level_index_path = os.path.join(self.index_dir, "second_stage_"+str(i))
            # # Load the index
            # second_level_faiss_index = faiss.read_index(second_level_index_path)
            # Read directly from mem
            second_level_faiss_index = self.second_level_faiss_indexes[i]
            num_docs = second_level_faiss_index.ntotal
            embedding_dimension = second_level_faiss_index.d
            new_embeddings = (faiss.rev_swig_ptr(second_level_faiss_index.get_xb(), num_docs*embedding_dimension).reshape(num_docs, embedding_dimension))
            new_embeddings = torch.tensor(new_embeddings)
            embeddings_list.append(new_embeddings)
        # Retrieve data from corpus
        documents = []
        for doc_id in doc_ids:
            record = self.corpus_list[doc_id]
            doc = Document(
                text=record["text"], metadata={"title": record["title"], "doc_id": record["_id"]}
            )
            documents.append(doc)
        # Create an online secondary level index
        faiss_index_second = faiss.IndexFlatL2(len(query_embedding[0]))
        for embeddings in embeddings_list:
            faiss_index_second.add(embeddings)
        # Look up with query embedding
        Dis, Idx = faiss_index_second.search(query_embedding, top_k)
        # Construct nodes with scores
        nodes_with_scores = []
        # Loop over returned indexes
        # Add and score context to response node
        for i, D in zip(Idx[0], Dis[0]):
           nodes_with_scores.append(NodeWithScore(node=documents[i], score=D))
        return nodes_with_scores
