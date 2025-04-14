import os
from shutil import rmtree
from typing import Callable, Dict, List, Optional
from llama_index.core.evaluation.benchmarks import BeirEvaluator
import tqdm
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.utils import get_cache_dir
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
import pickle
class BeirEvaluatorCustom(BeirEvaluator):
    def run(
        self,
        # create_retriever: Callable[[List[Document]], BaseRetriever],
        create_retriever: Callable[[VectorStoreIndex], BaseRetriever],
        datasets: List[str] = ["nfcorpus"],
        metrics_k_values: List[int] = [3],
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        k_retrieves = [1],
        k_evals = [1],
        chunk_size_large = 256,
        index_type = 'flatL2',
        num_embeddings = 768,
        ivf_num_cell = 128,
    ) -> None:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval

        dataset_paths = self._download_datasets(datasets)
        for (dataset, k_retrieve, k_eval) in zip(datasets, k_retrieves, k_evals):
            dataset_path = dataset_paths[dataset]
            print("Evaluating on dataset:", dataset)
            print("-------------------------------------")

            corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
                split="test"
            )
            # Load index here
            num_data = len(corpus.items())
            # if(num_data>=1024*128):
            #     ivf_num_cell=1024
            # else:
            #     ivf_num_cell=128
            index_dir = os.path.join(".", "index", dataset + "_" + str(chunk_size_large)+"_"+index_type)
            if(index_type=='ivfFlat'):
                index_dir = os.path.join(index_dir+"_"+str(ivf_num_cell))
            if(not os.path.exists(index_dir)):
                print("Index does not exist, skipping")
                continue
            # faiss_index = faiss.IndexFlatL2(num_embeddings)
            # vector_store = FaissVectorStore(faiss_index=faiss_index)
            # storage_context = StorageContext.from_defaults(vector_store=vector_store)

            vector_store = FaissVectorStore.from_persist_dir(index_dir)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=index_dir
            )
            index = load_index_from_storage(storage_context=storage_context)
            
            # documents = []
            # for id, val in corpus.items():
            #     doc = Document(
            #         text=val["text"], metadata={"title": val["title"], "doc_id": id}
            #     )
            #     documents.append(doc)
            retriever = create_retriever(index, k_retrieve)

            print("Retriever created for: ", dataset)

            print("Evaluating retriever on questions against qrels")

            results = {}
            for key, query in tqdm.tqdm(queries.items()):
                nodes_with_score = retriever.retrieve(query)
                node_postprocessors = node_postprocessors or []
                for node_postprocessor in node_postprocessors:
                    nodes_with_score = node_postprocessor.postprocess_nodes(
                        nodes_with_score, query_bundle=QueryBundle(query_str=query)
                    )
                results[key] = {
                    node.node.metadata["doc_id"]: node.score
                    for node in nodes_with_score
                }

            ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
                qrels, results, k_eval
            )
            print("Results for:", dataset)
            for k in k_eval:
                print(
                    {
                        f"NDCG@{k}": ndcg[f"NDCG@{k}"],
                        f"MAP@{k}": map_[f"MAP@{k}"],
                        f"Recall@{k}": recall[f"Recall@{k}"],
                        f"precision@{k}": precision[f"P@{k}"],
                    }
                )
            print("-------------------------------------")

    def run2(
        self,
        # create_retriever: Callable[[List[Document]], BaseRetriever],
        create_retriever: Callable,
        datasets: List[str] = ["nfcorpus"],
        metrics_k_values: List[int] = [3],
        nprobes = [1],
        k_retrieves = [1],
        k_evals = [[1]],
        cluster_size = 32,
        num_embeddings = 768,
        embed_model = None,

    ) -> None:
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval

        dataset_paths = self._download_datasets(datasets)
        for (dataset, k_retrieve, k_eval, nprobe) in zip(datasets, k_retrieves, k_evals, nprobes):
            dataset_path = dataset_paths[dataset]
            print("Evaluating on dataset:", dataset)
            print("-------------------------------------")

            corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
                split="test"
            )
            index_dir = os.path.join(".", "index", dataset + "_two_level")
            index_dir = os.path.join(index_dir+"_"+str(cluster_size))
            if(not os.path.exists(index_dir)):
                print("Index does not exist, skipping")
                continue
            # faiss_index = faiss.IndexFlatL2(num_embeddings)
            # vector_store = FaissVectorStore(faiss_index=faiss_index)
            # storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index_path = os.path.join(index_dir, "first_stage")
            index = faiss.read_index(index_path)
            redir_name = os.path.join(index_dir, "redir_table")
            redir_file = open(redir_name, "rb")
            redir_table = pickle.load(redir_file)
            redir_file.close()
            corpus_list = list(corpus.items())
            retriever = create_retriever(faiss_index=index, corpus_list=corpus_list, embed_model=embed_model, redir_table=redir_table, index_dir=index_dir)

            print("Retriever created for: ", dataset)

            print("Evaluating retriever on questions against qrels")

            results = {}
            for key, query in tqdm.tqdm(queries.items()):
                nodes_with_score = retriever.retrieve(query=query, nprobe=nprobe, top_k=k_retrieve)
                # node_postprocessors = node_postprocessors or []
                # for node_postprocessor in node_postprocessors:
                #     nodes_with_score = node_postprocessor.postprocess_nodes(
                #         nodes_with_score, query_bundle=QueryBundle(query_str=query)
                #     )
                results[key] = {
                    node.node.metadata["doc_id"]: node.score
                    for node in nodes_with_score
                }

            ndcg, map_, recall, precision = EvaluateRetrieval.evaluate(
                qrels, results, k_eval
            )
            print("Results for:", dataset)
            for k in k_eval:
                print(
                    {
                        f"NDCG@{k}": ndcg[f"NDCG@{k}"],
                        f"MAP@{k}": map_[f"MAP@{k}"],
                        f"Recall@{k}": recall[f"Recall@{k}"],
                        f"precision@{k}": precision[f"P@{k}"],
                    }
                )
            print("-------------------------------------")