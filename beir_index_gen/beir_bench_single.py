
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.evaluation.benchmarks import BeirEvaluator
from beir_evaluator_custom_index import BeirEvaluatorCustom
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from two_stage_retriever import TwoStageRetriever
from llama_index.core.retrievers import VectorIndexRetriever
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from two_stage_retriever_online import TwoStageRetrieverOnline
from two_stage_retriever_offline import TwoStageRetrieverOffline
import sys, argparse
Settings.chunk_size = 1024
Settings.chunk_overlap = 32
Settings.llm=None
embeddedModelName = "Alibaba-NLP/gte-base-en-v1.5"
embed_model = HuggingFaceEmbedding(model_name=embeddedModelName, trust_remote_code=True, embed_batch_size = 32, fp16=True)
Settings.embed_model = embed_model
def create_retriever(index, chunk_size_large=1024, chunk_overlap=32, chunk_size_small=256, chunk_overlap_small=32, two_stage=False, topk=4, topk_small=4):
    if(two_stage):
        r_engine = TwoStageRetriever(
            index=index,
            similarity_top_k=topk,
            similarity_top_k_second_stage = topk_small,
        )
    else:
        r_engine = VectorIndexRetriever(
            index=index,
            similarity_top_k=topk,
        )
    Settings.chunk_size = chunk_size_small
    Settings.chunk_overlap = chunk_overlap_small
    return r_engine

def create_retriever_small(index, k_retrieve):
    return create_retriever(index, chunk_size_large=256, chunk_overlap=32, chunk_size_small=256, chunk_overlap_small=32, topk=k_retrieve)

def create_retriever_large(index, k_retrieve):
    return create_retriever(index, chunk_size_large=1024, chunk_overlap=32, chunk_size_small=1024, chunk_overlap_small=32, topk=k_retrieve)

def create_retriever_two(index, k_retrieve):
    return create_retriever(index, chunk_size_large=1024, chunk_overlap=32, chunk_size_small=256, chunk_overlap_small=32, 
                            two_stage=True, topk=k_retrieve, topk_small=k_retrieve)

def create_two_level_retriever(faiss_index, corpus_list, redir_table, embed_model, index_dir):
    return TwoStageRetrieverOffline(faiss_index=faiss_index, corpus_list=corpus_list, redir_table=redir_table, embed_model=embed_model, index_dir=index_dir)





# datasets = ["nq"]
# print("Benchmarking small chunk")
# BeirEvaluatorCustom().run(
#     create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
# )
# BeirEvaluatorCustom().run(
#     create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="ivfFlat"
# )
parser = argparse.ArgumentParser(description = "Beir test driver")
parser.add_argument("--workload_idx", default=0)
parser.add_argument("--config_idx", default=0)
args = parser.parse_args()
workload_idx = int(args.workload_idx)
config_idx = int(args.config_idx)

datasets = ["scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever"]
k_retrieves = [6, 4, 3, 5, 5, 5]
k_evals = [[6], [4], [3], [5], [5], [5]]
nprobes=[2, 2, 2, 3, 3, 3]
cluster_sizes=[8,8,16,32,32,32]
datasets = [datasets[workload_idx]]
k_retrieves = [k_retrieves[workload_idx]]
k_evals = [k_evals[workload_idx]]
nprobes = [nprobes[workload_idx]]
cluster_size = cluster_sizes[workload_idx]
if(config_idx==0):
    print("Running small-block")
    BeirEvaluatorCustom().run(
        create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
    )
    print("Running small-block done!")
    print("-------------------------")
# elif(config_idx==1):
#     print("Running large-block")
#     BeirEvaluatorCustom().run(
#         create_retriever_large, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
#     )
#     print("Running large-block done!")
#     print("-------------------------")

# FAISS IVF and our implementation retrieve the same data chunks, tested...
# No need to run both

# elif(config_idx==2):
#     print("Running ivf")
#     BeirEvaluatorCustom().run(
#         create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="ivfFlat"
#     )
#     print("Running ivf done!")
#     print("-------------------------")
elif(config_idx==3):
    print("Running two-level, small")
    BeirEvaluatorCustom().run2(
        create_two_level_retriever, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, cluster_size=cluster_size, embed_model=embed_model, nprobes=nprobes,
    )
    print("Running two-level done!")
    print("-------------------------")

