import os
from shutil import rmtree
from typing import Callable, Dict, List, Optional

from tqdm import tqdm
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import Document, QueryBundle
from llama_index.core.utils import get_cache_dir
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import pickle
from llama_index.core.schema import TextNode, IndexNode
# import numpy as np
import time
import torch
# From beir
def download_datasets(datasets: List[str] = ["nfcorpus"]) -> Dict[str, str]:
    from beir import util

    cache_dir = get_cache_dir()

    dataset_paths = {}
    for dataset in datasets:
        dataset_full_path = os.path.join(cache_dir, "datasets", "BeIR__" + dataset)
        if not os.path.exists(dataset_full_path):
            url = f"""https://public.ukp.informatik.tu-darmstadt.de/thakur\
/BEIR/datasets/{dataset}.zip"""
            try:
                util.download_and_unzip(url, dataset_full_path)
            except Exception as e:
                print(
                    "Dataset:", dataset, "not found at:", url, "Removing cached dir"
                )
                rmtree(dataset_full_path)
                raise ValueError(f"invalid BEIR dataset: {dataset}") from e

        print("Dataset:", dataset, "downloaded at:", dataset_full_path)
        dataset_paths[dataset] = os.path.join(dataset_full_path, dataset)
    return dataset_paths



def runIndex(
    datasets: List[str] = ["nfcorpus"],
    embeddedModelName = "Alibaba-NLP/gte-base-en-v1.5",
    num_embeddings = 768,
    chunk_size_large = 1024,
    chunk_overlap = 32,
    index_type = "flatL2",
    ivf_num_cell = 128,
    ivf_train_limit = 1*1024*1024
) -> None:
    embed_model = HuggingFaceEmbedding(model_name=embeddedModelName, trust_remote_code=True, embed_batch_size = 32, fp16=True)
    Settings.embed_model = embed_model
    Settings.llm = None
    Settings.chunk_size = chunk_size_large
    Settings.chunk_overlap = chunk_overlap
    vector_store = None
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    dataset_paths = download_datasets(datasets)
    for dataset in datasets:
        dataset_path = dataset_paths[dataset]
        print("Indexing dataset:", dataset)
        print("-------------------------------------")
        index_dir = os.path.join(".", "index", dataset + "_" + str(chunk_size_large)+"_"+index_type)
        if(index_type=='ivfFlat'):
            index_dir = os.path.join(index_dir+"_"+str(ivf_num_cell))
        
        faiss_index = faiss.IndexFlatL2(num_embeddings)
        quantizer = faiss_index
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if(os.path.exists(index_dir)):
            print("Index exists, skipping")
            continue
        corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
            split="test"
        )
        documents = []
        cache_dir = get_cache_dir()
        num_data = len(corpus.items())
        counter = 0
        for id, val in corpus.items():
            doc = Document(
                text=val["text"], metadata={"title": val["title"], "doc_id": id}
            )
            documents.append(doc)
        if(index_type=="ivfFlat"):
            # quantizer = faiss_index
            faiss_index = faiss.IndexIVFFlat(quantizer, num_embeddings, ivf_num_cell)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Train
            print("Training quantizer!")
            text_group = []
            counter = 0
            for doc in documents:
                text = doc.get_text()
                text_group.append(text)
                counter += 1
                # 1M vectors should be enough for clustering.
                if(counter >= ivf_train_limit):
                    break
            embeddings = embed_model.get_text_embedding_batch(text_group)
            # Convert to Numpy Array
            # embeddings = np.array(embeddings)
            embeddings = torch.tensor(embeddings)
            # TODO: Convert?
            text_group = []
            faiss_index.train(embeddings)
            # print(faiss_index.clustering_index.centriods)
            # kmeans = faiss.Kmeans(768, 128, niter=20, verbose=True)
            # kmeans.train(embeddings)
            # # kmeans.centroids
            # # print(kmeans.centroids)
            # exit()
            print("IVF Quantizer training complete!")

        print("Indexing!")
        index = VectorStoreIndex.from_documents(
            documents = documents,
            show_progress=True,
            # May need to tune this again?
            insert_batch_size=2048,
            storage_context = storage_context,
            chunk_size = chunk_size_large,
            chunk_overlap = chunk_overlap,
        )
        print("Indexing done! Persisting the index.")
        # Persist the index
        index.storage_context.persist(persist_dir=index_dir)
        print("Created and persisted the index")            



def runIndex_first_stage(
    datasets: List[str] = ["nfcorpus"],
    embeddedModelName = "Alibaba-NLP/gte-base-en-v1.5",
    num_embeddings = 768,
    chunk_size_large = 1024,
    chunk_overlap = 32,
    index_type = "flatL2",
    cluster_train_limit = 1*1024*1024,
    cluster_size = 128,
    kmeans_niter = 20,
) -> None:
    embed_model = HuggingFaceEmbedding(model_name=embeddedModelName, trust_remote_code=True, embed_batch_size = 32, fp16=True)
    Settings.embed_model = embed_model
    Settings.llm = None
    Settings.chunk_size = chunk_size_large
    Settings.chunk_overlap = chunk_overlap
    vector_store = None
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    dataset_paths = download_datasets(datasets)
    for dataset in datasets:
        dataset_path = dataset_paths[dataset]
        print("Indexing dataset:", dataset)
        print("-------------------------------------")
        index_dir = os.path.join(".", "index", dataset + "_two_level")
        index_dir = os.path.join(index_dir+"_"+str(cluster_size))
        if(os.path.exists(index_dir)):
            print("Index exists, skipping")
            continue
        
        corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(
            split="test"
        )
        # Not really need here? Python 3.7+ guarantee ordering?
        corpus_list = list(corpus.items())
        documents = []
        cache_dir = get_cache_dir()
        num_data = len(corpus_list)
        num_cluster = num_data/cluster_size
        # Should not be necessary, but just in case.
        if(num_cluster<1):
            print("Invalid number of cluster, using 1.")
            num_cluster = 1
        
        # Create FAISS FlatL2 index
        faiss_index = faiss.IndexFlatL2(num_embeddings)
        counter = 0
        for id, val in corpus_list:
            doc = Document(
                text=val["text"], metadata={"title": val["title"], "doc_id": id}
            )
            documents.append(doc)
        text_group = []
        counter = 0
        skip_ratio = num_data//cluster_train_limit
        if(skip_ratio<1):
            skip_ratio = 1
        for doc in documents:
            # # Subsampling for clustering
            # if(counter%skip_ratio != 0):
            #     counter += 1
            #     continue
            text = doc.get_text()
            text_group.append(text)
            counter += 1
        print("Kmeans samples: " + str(len(text_group)))
        print("Generating embeddings")
        embeddings = embed_model.get_text_embedding_batch(text_group, show_progress=True)
        # Convert to Numpy Array
        # embeddings = np.array(embeddings, dtype=np.float16)
        embeddings = torch.tensor(embeddings)
        # TODO: Convert?
        print(embeddings.dtype)
        print("Running Kmeans!")
        start = time.time()
        kmeans = faiss.Kmeans(num_embeddings, num_cluster, niter=kmeans_niter, verbose=True)
        kmeans.train(embeddings)
        stop = time.time()
        kmeans_time = stop - start
        print("Kmeans time: " + str(kmeans_time))
        # Convert centroids to FP16
        centroids = torch.tensor(kmeans.centroids)
        # TODO: Convert?
        # centroids = np.float16(centroids)
        # Add centroids to first stage flatL2 index
        faiss_index.add(centroids)
        # Adding data to redirection table
        redir_table = {}
        counter = 0
        # Create a list for each centroid
        for centroid in kmeans.centroids:
            redir_table[counter] = list()
            counter += 1        
        # Add document indexes to each centroid
        counter = 0
        batch_counter = 0
        # TODO: Convert?
        # Search index and return nearest clusters
        Dis, Idx = faiss_index.search(embeddings, 1)
        corpus_index = 0
        for indexes in Idx:
            index = indexes[0]
            redir_table[index] += [corpus_index]
            corpus_index += 1
        counter += batch_counter
        batch_counter = 0
        text_group = []
        counter = 0
        length_list = []
        print("Centroids stats")
        for centroid in redir_table.keys():
            print("Centroid " + str(counter) + ": " + str(len(redir_table[centroid])))
            length_list += [len(redir_table[centroid])]
            counter += 1

        print("Persisting the index.")
        # Create a dir
        os.mkdir(index_dir)
        # Persist the index
        index_path = os.path.join(index_dir, "first_stage")
        faiss.write_index(faiss_index, index_path)
        # Persist redirection table
        redir_name = os.path.join(index_dir, "redir_table")
        redir_file = open(redir_name, "wb")
        pickle.dump(redir_table, redir_file)
        redir_file.close()
        print("Created and persisted the index")
        # print("Testing")
        # index2 = faiss.read_index(index_path)
        # redir_file2 = open(redir_name, "rb")
        # redir2 = pickle.load(redir_file2)
        # print(str(index2))
        # print(str(redir2))




def create_second_stage(
    datasets: List[str] = ["nfcorpus"],
    embeddedModelName = "Alibaba-NLP/gte-base-en-v1.5",
    num_embeddings = 768,
    chunk_size_large = 1024,
    chunk_overlap = 32,
    index_type = "flatL2",
    cluster_train_limit = 1*1024*1024,
    cluster_size = 128,
    kmeans_niter = 20,
) -> None:
    embed_model = HuggingFaceEmbedding(model_name=embeddedModelName, trust_remote_code=True, embed_batch_size = 256, fp16=True)
    Settings.embed_model = embed_model
    Settings.llm = None
    Settings.chunk_size = chunk_size_large
    Settings.chunk_overlap = chunk_overlap
    vector_store = None
    from beir.datasets.data_loader import GenericDataLoader
    from beir.retrieval.evaluation import EvaluateRetrieval
    from llama_index.core.evaluation.benchmarks import BeirEvaluator
    import datasets as ds
    for dataset in datasets:
        print("Indexing dataset:", dataset)
        print("-------------------------------------")
        index_dir = os.path.join(".", "index", dataset + "_two_level")
        index_dir = os.path.join(index_dir+"_"+str(cluster_size))
        if(not os.path.exists(index_dir)):
            print("Index does not exist, skipping")
            continue
        second_dir = os.path.join(index_dir, "gen_cost_table")
        if(os.path.exists(second_dir)):
            print("Second-level index created, skipping")
            continue
        # DO NOT LOAD ALL DATA AT ONCE!
        BE = BeirEvaluator()
        dataset_paths = BE._download_datasets(datasets=[dataset])
        ds_path = dataset_paths[dataset]
        corpus_sets = ds.load_dataset(path=ds_path, data_files='corpus.jsonl', keep_in_memory=False)
        corpus = corpus_sets['train']
        redir_name = os.path.join(index_dir, "redir_table")
        redir_file = open(redir_name, "rb")
        redir_table = pickle.load(redir_file)
        redir_file.close()
        
        # Embed gen Cost table
        embed_gen_costs = {}
        # For each centroid, pull all docs and evaluate the cost of embedding generation
        for centroid_idx in tqdm(redir_table.keys()):
            # Fetch all docs
            # docs = []
            text_group = []
            centroid_index_cost = 0
            # Skip empty centroids
            if(redir_table[centroid_idx]==0):
                continue
            for doc_idx in redir_table[centroid_idx]:
                # retrieve doc
                unformatted_doc = corpus[doc_idx]
                # doc = Document(
                #     text=unformatted_doc["text"], metadata={"title": unformatted_doc["title"], "doc_id": unformatted_doc["_id"]}
                # )
                # docs+=doc
                text_group.append(unformatted_doc["text"])
                # Update cost
                centroid_index_cost += len(unformatted_doc["text"])
            # Generate embeddings
            centroid_embeddings = embed_model.get_text_embedding_batch(text_group)
            centroid_embeddings = torch.tensor(centroid_embeddings)
            # Create a second-level index here
            # We have checked if the index is not empty.
            embed_gen_costs[centroid_idx] = centroid_index_cost
            faiss_index_second = faiss.IndexFlatL2(len(centroid_embeddings[0]))
            # Add embeddings to second-level cluster
            faiss_index_second.add(centroid_embeddings)
            # Persist the index
            index_path = os.path.join(index_dir, "second_stage_"+str(centroid_idx))
            faiss.write_index(faiss_index_second, index_path)
            # Persist embed gen cost table
            cost_table_name = os.path.join(index_dir, "gen_cost_table")
            cost_table_file = open(cost_table_name, "wb")
            pickle.dump(embed_gen_costs, cost_table_file)
            cost_table_file.close()

datasets = ["nfcorpus", "scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever", ]
# Create baseline Flat index
runIndex(datasets = datasets, chunk_size_large=256, index_type="flatL2")
# We are not creating IVF index, we use the same config for baseline but we load all embeddings in the memory.
# runIndex(datasets = datasets, chunk_size_large=1024, index_type="flatL2")
# runIndex(datasets = datasets, chunk_size_large=256, index_type="ivfFlat", ivf_num_cell=128)

# Create index with pre-compute embeddings
# Small cluster + large dataset may take too long to cluster!

runIndex_first_stage(datasets, chunk_size_large=256, kmeans_niter=15, cluster_size = 32)
runIndex_first_stage(datasets, chunk_size_large=256, kmeans_niter=15, cluster_size = 64)
runIndex_first_stage(datasets, chunk_size_large=256, kmeans_niter=10, cluster_size = 128)


create_second_stage(datasets=datasets, cluster_size = 8)
create_second_stage(datasets=datasets, cluster_size = 16)
create_second_stage(datasets=datasets, cluster_size = 32)

