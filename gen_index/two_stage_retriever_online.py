
from llama_index.core import Document, Settings
from typing import Any, List, Optional, Dict
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.embeddings.base import BaseEmbedding
import faiss
import torch
# Custom retriever compatible with BEIR, not a drop-in replacement for VectorIndexRetriever
class TwoStageRetrieverOnline():
    def __init__(self,
        faiss_index: callable,
        corpus_list: List,
        redir_table: Dict,
        embed_model: Optional[BaseEmbedding] = None,
    ) -> None:
        self.faiss_index = faiss_index
        self.corpus_list = corpus_list
        self.embed_model = embed_model
        self.redir_table = redir_table
    
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
        for i in Idx[0]:
            doc_ids += self.redir_table[i]

        # Retrieve data from corpus
        documents = []
        for doc_id in doc_ids:
            record = self.corpus_list[doc_id]
            id, val = record
            doc = Document(
                text=val["text"], metadata={"title": val["title"], "doc_id": id}
            )
            documents.append(doc)
        # Create an online secondary level index
        faiss_index_second = faiss.IndexFlatL2(len(query_embedding[0]))
        candidate_embeddings = []
        text_group = []
        for doc in documents:
            text = doc.get_text()
            text_group.append(text)
        embeddings = self.embed_model.get_text_embedding_batch(text_group)
        # Convert Embeddings to FP16
        embeddings = torch.tensor(embeddings)
        # Insert embeddins into an index
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

