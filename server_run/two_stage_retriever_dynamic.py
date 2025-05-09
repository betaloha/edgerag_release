
from llama_index.core import Document, Settings
from typing import Any, List, Optional, Dict
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
import faiss
import torch
import os
import numpy as np
# Custom retriever compatible with BEIR, not a drop-in replacement for VectorIndexRetriever
class TwoStageRetrieverDynamic():
    def __init__(self,
        faiss_index: callable,
        corpus_list: List,
        redir_table: Dict,
        cost_table: Dict,
        index_dir: str,
        embed_model: Optional[BaseEmbedding] = None,
        max_probe_cost: Optional[int] = -1,
    ) -> None:
        self.faiss_index = faiss_index
        self.corpus_list = corpus_list
        self.embed_model = embed_model
        self.index_dir = index_dir
        self.redir_table = redir_table
        self.cost_table = cost_table
        self.total_generate = 0
        self.total_load = 0
        self.max_probe_cost = max_probe_cost
    def retrieve(
        self,
        query: str,
        top_k: int,
        nprobe: int,
        # MAX_COST_PER_PROBE: Optional[int] = -1,

    ) -> List[NodeWithScore]:
        # Compute auto MAX_COST with retrieve max time = 500ms
        if(self.max_probe_cost<=0):
            self.max_probe_cost = 30000
        print("TOTAL_PROBE_COST: " + str(self.max_probe_cost))
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
        documents = []
        for i in Idx[0]:
            doc_ids = self.redir_table[i]
            text_group = []
            for doc_id in doc_ids:
                record = self.corpus_list[doc_id]
                doc = Document(
                    text=record["text"], metadata={"title": record["title"], "doc_id": record["_id"]}
                )
                documents.append(doc)
                text_group.append(record["text"])
            if(self.cost_table[i] > int(self.max_probe_cost/nprobe)):
                print("Loading")
                # Retrieve Second-level index
                second_level_index_path = os.path.join(self.index_dir, "second_stage_"+str(i))
                # Load the index
                second_level_faiss_index = faiss.read_index(second_level_index_path)
                num_docs = second_level_faiss_index.ntotal
                embedding_dimension = second_level_faiss_index.d
                new_embeddings = (faiss.rev_swig_ptr(second_level_faiss_index.get_xb(), num_docs*embedding_dimension).reshape(num_docs, embedding_dimension))
                new_embeddings = torch.tensor(new_embeddings)
                embeddings_list.append(new_embeddings)
                self.total_load +=1
            else:
                print("Generating")
                new_embeddings = self.embed_model.get_text_embedding_batch(text_group)
                new_embeddings = torch.tensor(new_embeddings)
                embeddings_list.append(new_embeddings)
                self.total_generate += 1
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

    def getStats(self):
        return (self.total_generate, self.total_load)
