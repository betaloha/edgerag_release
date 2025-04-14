
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.core.evaluation.benchmarks import BeirEvaluator
from beir_evaluator_custom_index import BeirEvaluatorCustom
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from two_stage_retriever import TwoStageRetriever
from llama_index.core.retrievers import VectorIndexRetriever
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from two_stage_retriever_online import TwoStageRetrieverOnline

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

def create_two_level_retriever(faiss_index, corpus_list, redir_table, embed_model):
    return TwoStageRetrieverOnline(faiss_index=faiss_index, corpus_list=corpus_list, redir_table=redir_table, embed_model=embed_model)


k_vals = [1, 2, 3, 4]
# datasets = ["nfcorpus", "scifact", "msmarco", "nq", "hotpotqa", "cqadupstack", "quora", "scidocs"]
# k_retrieves = [1,1,1,1,2,2,2,5]
# k_evals = [1,1,1,1,2,2,2,5]
datasets = ["nfcorpus", "scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever"]
k_retrieves = [1,1,1,1,1,1,1]
k_evals = [[1],[1],[1],[1],[1],[1],[1]]
nprobes=[4,4,4,4,4,4,4]

k_retrieves = [1]
k_evals = [[1]]
nprobes=[1]
datasets = ["quora"]


# datasets = ["nq"]
# print("Benchmarking small chunk")
# BeirEvaluatorCustom().run(
#     create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
# )
# BeirEvaluatorCustom().run(
#     create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="ivfFlat"
# )
k_retrieves = [5, 2, 2, 1, 2, 2]
k_evals = [[5], [2], [2], [1], [2], [2]]
nprobes=[2, 2, 2, 2, 2, 2]
datasets = ["scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever"]
print("Running small-block")
BeirEvaluatorCustom().run(
    create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
)
print("Running small-block done!")
print("-------------------------")
print("Running large-block")
BeirEvaluatorCustom().run(
    create_retriever_large, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
)
print("Running large-block done!")
print("-------------------------")
print("Running ivf")
BeirEvaluatorCustom().run(
    create_retriever_small, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, index_type="ivfFlat"
)
print("Running ivf done!")
print("-------------------------")
print("Running two-level, small")
k_retrieves = [5, 2]
k_evals = [[5], [2]]
nprobes=[2, 2]
datasets = ["scidocs", "fiqa"]
BeirEvaluatorCustom().run2(
    create_two_level_retriever, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, cluster_size=8, embed_model=embed_model, nprobes=nprobes,
)
k_retrieves = [2, 1, 2, 2]
k_evals = [[2], [1], [2], [2]]
nprobes=[2, 2, 2, 2]
datasets = ["quora", "nq", "hotpotqa", "fever"]
print("Running two-level done!")
print("-------------------------")
# k_retrieves = [1, 1, 1, 1]
# k_evals = [1, 1, 1, 1]
# nprobes=[1,2,3,4]
# datasets = ["nq", "nq", "nq", "nq"]
# BeirEvaluatorCustom().run2(
#     create_two_level_retriever, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, cluster_size=32, embed_model=embed_model, nprobes=nprobes,
# )

# k_retrieves = [2, 2, 2, 2]
# k_evals = [2, 2, 2, 2]
# nprobes=[1,2,3,4]
# datasets = ["hotpotqa", "hotpotqa", "hotpotqa", "hotpotqa"]
# BeirEvaluatorCustom().run2(
#     create_two_level_retriever, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, cluster_size=32, embed_model=embed_model, nprobes=nprobes,
# )



# print("Benchmarking large chunk")
# BeirEvaluator().run(
#     create_retriever_large, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals
# )
# print("Benchmarking two-level")
# BeirEvaluator().run(
#     create_retriever_two, datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals
# )
