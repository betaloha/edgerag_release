import sys, argparse, os
parser = argparse.ArgumentParser(description = "Beir test driver")
parser.add_argument("--workload_idx", default=0)
parser.add_argument("--config_idx", default=0)
parser.add_argument("--llm_model", default="princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
parser.add_argument("--llm_evaluation", action=argparse.BooleanOptionalAction)
args = parser.parse_args()
workload_idx = int(args.workload_idx)
config_idx = int(args.config_idx)
llm_model = args.llm_model
debug_flag = bool(args.debug)
llm_evaluation = bool(args.llm_evaluation)

datasets = ["scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever"]
k_retrieves = [5, 2, 2, 1, 2, 2]
k_evals = [[5], [2], [2], [1], [2], [2]]
nprobes=[2, 2, 2, 3, 3, 3]
cluster_sizes=[8,8,16,32,32,32]
datasets = [datasets[workload_idx]]
k_retrieves = [k_retrieves[workload_idx]]
k_evals = [k_evals[workload_idx]]
nprobes = [nprobes[workload_idx]]
cluster_size = cluster_sizes[workload_idx]


from llama_index.core import StorageContext, load_index_from_storage, Settings
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from beir.datasets.data_loader import GenericDataLoader
from typing import Dict
from query_loader import QueryOnlyDataLoader
import time
from nano_llm import NanoLLM

def create_retriever(index, topk):
    r_engine = VectorIndexRetriever(
        index=index,
        similarity_top_k=topk,
    )
    return r_engine



def llm_eval_run(
    dataset="nfcorpus",
    chunk_size=256,
    k_retrieve=1,
    index_type="flatL2",
    ivf_num_cell=128,
    query_limit=10,
    ):
    from llama_index.core.base.response.schema import Response
    # Load LLM
    model = NanoLLM.from_pretrained(model=llm_model, api='mlc', quantization='q4f16_ft')
    # Disable LLamaindex default LLM
    Settings.llm = None
    # Load Embedding model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, embed_batch_size = 32, fp16=True)
    # Load index
    index_dir = os.path.join(".", "index", dataset + "_" + str(chunk_size)+"_"+index_type)
    if(index_type=='ivfFlat'):
        index_dir = os.path.join(index_dir+"_"+str(ivf_num_cell))
    if(not os.path.exists(index_dir)):
        print("Index does not exist, exiting")
        exit()
    # Load vectorstore
    vector_store = FaissVectorStore.from_persist_dir(index_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=index_dir
    )
    # Also need to load corpus from docstore
    index = load_index_from_storage(storage_context=storage_context)
    retriever = create_retriever(index = index, topk = k_retrieve)
    
    # LLM Evaluation
    gpt4 = None
    evaluator_gpt4 = None
    if(llm_evaluation):
        from llama_index.llms.openai import OpenAI
        from llama_index.core.evaluation import RelevancyEvaluator
        gpt4 = OpenAI(temperature=0, model="gpt-4o")
        evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

    dataset_paths = BeirEvaluator._download_datasets([datasets])
    # Load queries
    queries = QueryOnlyDataLoader.load(data_folder=[dataset_paths])
    query_counter = 0
    relevancy_score_total = 0
    retrieve_time_total = 0
    first_token_gen_time_total = 0
    generation_time_total = 0
    for query in queries:
        # Retrieve
        start_retrieve = time.time()
        retrieved_nodes = retriever.retrieve(query)
        # Convert Nodes to plain text
        retrieved_text = ""
        for node in retrieved_nodes:
            retrieved_text = retrieved_text + node.get_text() + " "
        # Prepare the prompt
        question = str(query)
        prompt="We have provided context information below. \n"+\
            "---------------------\n"+\
            retrieved_text+\
            "\n---------------------\n"+\
            "Given this information, do not repeat the answers, if given how many or who is question, give only one answer, please give short, clear and precise answer to the question below \n" +\
            "\n---------------------\n"+ question+"\n"
        # Generate
        generation_start_time = time.time()
        response = model.generate(prompt, max_new_tokens=296, streaming=True)
        response_str = ""
        is_first_token = True
        if debug_flag:
            print("Response:")
        token_count = 0
        for token in response:
            response_str += str(token)
            token_count = token_count + 1
            if debug_flag:
                print(token, end='\n\n' if response.eos else '', flush=True)
            if(token != None):
                pass
            if(is_first_token):
                first_token_time = time.time()
                is_first_token = False
        print("")
        generation_end_time = time.time()
        llm_score = 0
        # LLM Evaluation
        if(llm_evaluation):
            response_vector = Response(response=response_str, source_nodes=retrieved_nodes, )
            eval_result = evaluator_gpt4.evaluate_response(
                query=question, response=response_vector)
            relevancy_score_total += eval_result.score
            print("E-STATS, Passing: " + str(eval_result.passing) + ", Score: " + (str(relevancy_score_total)))
        # Computing latency
        retrieve_time = generation_start_time - start_retrieve
        first_token_gen_time = first_token_time - generation_start_time
        generation_time = generation_end_time - generation_start_time
        retrieve_time_total += retrieve_time
        first_token_gen_time_total += first_token_gen_time
        generation_time_total += generation_time
        print("Q_STATS, Retrieve Time: " + str(retrieve_time) + ", Prefill Latency: " + str(first_token_gen_time) + ", Generation Latency: " + str(generation_time) + ", Generated Tokens: " + str(token_count))
        query_counter += 1
        if(query_counter>=query_limit):
            break
    print("S_STATS, Dataset: " + dataset + ", Config: " + index_type + ", chunk_size: " + str(chunk_size) + ", Evaluated_Query: " + str(query_counter) + ", Total_Score: " + str(relevancy_score_total) + \
          ", Retrieve_time_total: " + str(retrieve_time_total) + ", First_token_gen_time_total: " + str(first_token_gen_time_total) + ", Generation_time_total: " + str(generation_time_total))


def llm_eval_run2(
    dataset="nfcorpus",
    chunk_size=256,
    k_retrieve=1,
    index_type="flatL2",
    nprobes=1,
    query_limit=10,
    ):
    pass


if(config_idx==0):
    print("Running small-block")
    llm_eval_run(
        datasets=datasets, chunk_size = 256, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
    )
    print("Running small-block done!")
    print("-------------------------")
elif(config_idx==1):
    print("Running large-block")
    llm_eval_run(
        datasets=datasets, chunk_size = 1024, k_retrieves=k_retrieves, k_evals=k_evals, index_type="flatL2"
    )
    print("Running large-block done!")
    print("-------------------------")
elif(config_idx==2):
    print("Running ivf")
    llm_eval_run(
        datasets=datasets, chunk_size = 256, k_retrieves=k_retrieves, k_evals=k_evals, index_type="ivfFlat"
    )
    print("Running ivf done!")
    print("-------------------------")
elif(config_idx==3):
    print("Running two-level, small")
    llm_eval_run2(
        datasets=datasets, k_retrieves=k_retrieves, k_evals=k_evals, cluster_size=cluster_size, nprobes=nprobes,
    )
    print("Running two-level done!")
    print("-------------------------")