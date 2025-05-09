import sys, argparse, os
parser = argparse.ArgumentParser(description = "Beir test driver")
parser.add_argument("--workload_idx", default=0)
parser.add_argument("--config_idx", default=0)
parser.add_argument("--llm_model", default="princeton-nlp/Sheared-LLaMA-2.7B-ShareGPT")
parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
parser.add_argument("--llm_evaluation", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--num_query", default=10)
parser.add_argument("--num_skip", default=0)
parser.add_argument("--no_generation", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--load_cache", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--max_probe_cost", default=-1)
parser.add_argument("--cache_max_size", default=-1)
parser.add_argument("--min_cache_cost", default=0)

parser.add_argument("--save_cache_checkpoint", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--load_cache_checkpoint", action=argparse.BooleanOptionalAction, default=False)

args = parser.parse_args()
workload_idx = int(args.workload_idx)
config_idx = int(args.config_idx)
llm_model = args.llm_model
debug_flag = bool(args.debug)
llm_evaluation = bool(args.llm_evaluation)
num_query = int(args.num_query)
num_skip = int(args.num_skip)
no_generation = bool(args.no_generation)
prefetch = bool(args.prefetch)
load_cache = bool(args.load_cache)
max_probe_cost = int(args.max_probe_cost)
cache_max_size = int(args.cache_max_size)
save_cache_checkpoint = bool(args.save_cache_checkpoint)
load_cache_checkpoint = bool(args.load_cache_checkpoint)
min_cache_cost = int(args.min_cache_cost)
# override generation
if(prefetch):
    no_generation = True
import datasets as ds
datasets = ["scidocs", "fiqa", "quora", "nq", "hotpotqa", "fever"]
if(config_idx>=3):
    k_retrieves = [6, 4, 3, 5, 5, 5]
    k_evals = [[6], [4], [3], [5], [5], [5]]
    nprobes=[2, 2, 2, 3, 3, 3]
    cluster_sizes=[8,8,16,32,32,32]
else:
    k_retrieves = [5, 2, 2, 2, 2, 2]
    k_evals = [[5], [2], [2], [2], [2], [2]]
    # Not used
    nprobes=[2, 2, 2, 4, 4, 4] 
    cluster_sizes=[8,8,16,32,32,32]
dataset = datasets[workload_idx]
k_retrieve = k_retrieves[workload_idx]
nprobe = nprobes[workload_idx]
cluster_size = cluster_sizes[workload_idx]
query_limit = [1000, 648, 10000, 3452, 7405, 6666]
if(num_query == -1):
    num_query = query_limit[workload_idx]
p95_cost = [21990, 18318, 2348, 36665, 22861, 55296]
p90_cost = [18034, 14020, 1925, 29836, 17909, 38710]
SLO_cost = [30000, 30000, 30000, 30000, 30000, 30000]
if(max_probe_cost == -1):
    max_probe_cost = SLO_cost[workload_idx]*nprobes[workload_idx]
print("Using max probe cost of " + str(max_probe_cost))
from llama_index.core import StorageContext, load_index_from_storage, Settings
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from beir.datasets.data_loader import GenericDataLoader
from typing import Dict
from query_loader import QueryOnlyDataLoader
import time

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
    skip=0,
    ):
    from llama_index.core.base.response.schema import Response
    # Load LLM
    model = None
    if(not no_generation):
        from nano_llm import NanoLLM
        model = NanoLLM.from_pretrained(model=llm_model, api='mlc', quantization='q4f16_ft')
        # For long context length
        model.tokenizer.model_max_length=4096
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
    # Load queries
    BE = BeirEvaluator()
    dataset_paths = BE._download_datasets(datasets=[dataset])
    ds_path = dataset_paths[dataset]
    query_sets = ds.load_dataset(path=ds_path, data_files='queries.jsonl', keep_in_memory=False)
    queries = query_sets['train']
    # print(queries)
    # exit()
    query_counter = 0
    relevancy_score_total = 0
    retrieve_time_total = 0
    first_token_gen_time_total = 0
    generation_time_total = 0
    for query_entry in queries:
        query = query_entry['text']
        if(skip>0):
            skip -= 1
            continue
        # Retrieve
        start_retrieve = time.time()
        first_token_time = start_retrieve
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
        response = []
        if(no_generation==False):
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
            relevancy_score = eval_result.score
            relevancy_score_total += relevancy_score
            print("E-STATS, Passing: " + str(eval_result.passing) + ", Score: " + (str(relevancy_score)))
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
    nprobe=1,
    query_limit=10,
    skip=0,
    config_idx = 3
    ):
    from llama_index.core.base.response.schema import Response
    # Load LLM
    model=None
    if(not no_generation):
        from nano_llm import NanoLLM
        model = NanoLLM.from_pretrained(model=llm_model, api='mlc', quantization='q4f16_ft')
        # For long context length
        model.tokenizer.model_max_length=4096
    
    # Disable LLamaindex default LLM
    Settings.llm = None
    # Load Embedding model
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    embed_model = HuggingFaceEmbedding(model_name="Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True, embed_batch_size = 32, fp16=True)
    Settings.embed_model = embed_model
    # Load index
    import pickle
    index_dir = os.path.join(".", "index", dataset + "_two_level")
    index_dir = os.path.join(index_dir+"_"+str(cluster_size))
    if(not os.path.exists(index_dir)):
        print("Index does not exist, skipping")
        exit()

    index_path = os.path.join(index_dir, "first_stage")
    index = faiss.read_index(index_path)
    redir_name = os.path.join(index_dir, "redir_table")
    redir_file = open(redir_name, "rb")
    redir_table = pickle.load(redir_file)
    redir_file.close()
    cost_table_name = os.path.join(index_dir, "gen_cost_table")
    cost_table_file = open(cost_table_name, "rb")
    cost_table = pickle.load(cost_table_file)
    cost_table_file.close()
    # Load Cache
    cache_state = None
    if(load_cache):
        cache_path = os.path.join(".", "cache")
        cache_dict_name = os.path.join(cache_path, "cache_" + str(workload_idx)+ "_" + str(max_probe_cost))
        if(not os.path.exists(cache_dict_name)):
            print("Invalid cache dir, exiting")
            exit()
        cache_dict_file = open(cache_dict_name, "rb")
        cache_state = pickle.load(cache_dict_file)
        cache_dict_file.close()
    if(load_cache_checkpoint):
        cache_path = os.path.join(".", "cache_temp")
        cache_dict_name = os.path.join(cache_path, "cache_" + str(workload_idx)+ "_" + str(cache_max_size))
        if(not os.path.exists(cache_dict_name)):
            print("Invalid cache dir, exiting")
            exit()
        cache_dict_file = open(cache_dict_name, "rb")
        cache_state = pickle.load(cache_dict_file)
        cache_dict_file.close()
    # Load corpus and queries
    BE = BeirEvaluator()
    dataset_paths = BE._download_datasets(datasets=[dataset])
    ds_path = dataset_paths[dataset]
    corpus_sets = ds.load_dataset(path=ds_path, data_files='corpus.jsonl', keep_in_memory=False)
    corpus = corpus_sets['train']
    # query_sets = ds.load_dataset(path=ds_path, data_files='queries.jsonl', keep_in_memory=False)
    # queries = query_sets['train']
    queries = QueryOnlyDataLoader(data_folder=ds_path).load_only_queries()
    # Create a retriever
    retriever = None
    if(config_idx == 3):
        from two_stage_retriever_online import TwoStageRetrieverOnline
        retriever = TwoStageRetrieverOnline(faiss_index=index, corpus_list=corpus, 
            embed_model=embed_model, redir_table=redir_table)
    elif(config_idx == 4):
        from two_stage_retriever_offline import TwoStageRetrieverOffline
        retriever = TwoStageRetrieverOffline(faiss_index=index, corpus_list=corpus, 
            embed_model=embed_model, redir_table=redir_table, index_dir=index_dir)
    elif(config_idx == 5):
        from two_stage_retriever_dynamic import TwoStageRetrieverDynamic
        retriever = TwoStageRetrieverDynamic(faiss_index=index, corpus_list=corpus, 
            embed_model=embed_model, redir_table=redir_table, index_dir=index_dir, cost_table=cost_table, max_probe_cost = max_probe_cost)
    elif(config_idx == 6):
        from two_stage_retriever_dynamic_cache_LRU import TwoStageRetrieverDynamicCached
        if(load_cache):
            # First Run
            retriever = TwoStageRetrieverDynamicCached(faiss_index=index, corpus_list=corpus, 
                embed_model=embed_model, redir_table=redir_table, index_dir=index_dir, 
                cost_table=cost_table, cache_obj=cache_state, max_probe_cost=max_probe_cost, 
                cache_max_size=cache_max_size, min_cache_cost = min_cache_cost)
        elif(load_cache_checkpoint):
            # Subsequent Runs, load checkpoint
            retriever = TwoStageRetrieverDynamicCached(faiss_index=index, corpus_list=corpus, 
                embed_model=embed_model, redir_table=redir_table, index_dir=index_dir, 
                cost_table=cost_table, cache_obj=cache_state, max_probe_cost=max_probe_cost, 
                cache_max_size=cache_max_size, min_cache_cost = min_cache_cost)
        else:
            # Prefetch
            retriever = TwoStageRetrieverDynamicCached(faiss_index=index, corpus_list=corpus, 
                embed_model=embed_model, redir_table=redir_table, index_dir=index_dir, 
                cost_table=cost_table, cache_max_size=-1, max_probe_cost=max_probe_cost, min_cache_cost = min_cache_cost)
    else:
        print("Incorrect config, exiting")
        exit()

    # LLM Evaluation
    gpt4 = None
    evaluator_gpt4 = None
    if(llm_evaluation):
        from llama_index.llms.openai import OpenAI
        from llama_index.core.evaluation import RelevancyEvaluator
        gpt4 = OpenAI(temperature=0, model="gpt-4o")
        evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

    # print(queries)
    # exit()
    query_counter = 0
    relevancy_score_total = 0
    retrieve_time_total = 0
    first_token_gen_time_total = 0
    generation_time_total = 0
    # for query_entry in queries:
    for key, query in queries.items():
        # query = query_entry['text']
        if(skip>0):
            skip -= 1
            continue
        # Retrieve
        start_retrieve = time.time()
        first_token_time = start_retrieve
        retrieved_nodes = retriever.retrieve(query, top_k=k_retrieve, nprobe=nprobe)
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
        response = []
        if(no_generation==False):
            if(len(prompt)>8192):
                continue
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
            relevancy_score = eval_result.score
            relevancy_score_total += relevancy_score
            print("E-STATS, Passing: " + str(eval_result.passing) + ", Score: " + (str(relevancy_score)))
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
    if(config_idx>=5):
        print("G_stats, Load: " + str(retriever.total_load) + ", Generate: " + str(retriever.total_generate))
    if(config_idx>=6):
        print("C_stats, Hit: " + str(retriever.total_cache_hit) + ", Missed: " + str(retriever.total_cache_miss) + ", NC: " + str(retriever.total_non_cache))
    print("S_STATS, Dataset: " + dataset + ", Config: " + "two-nprobe=" + str(nprobe) + "-retrive=" + str(k_retrieve) + ", chunk_size: " + str(chunk_size) + ", Evaluated_Query: " + str(query_counter) + ", Total_Score: " + str(relevancy_score_total) + \
          ", Retrieve_time_total: " + str(retrieve_time_total) + ", First_token_gen_time_total: " + str(first_token_gen_time_total) + ", Generation_time_total: " + str(generation_time_total))
    # Save cache object
    if(prefetch):
        cache_path = os.path.join(".", "cache")
        os.makedirs(cache_path, exist_ok=True)
        cache_dict_name = os.path.join(cache_path, "cache_" + str(workload_idx)+ "_" + str(max_probe_cost))
        cache_dict_file = open(cache_dict_name, "wb")
        pickle.dump(retriever.get_cache_dict(), cache_dict_file)
        cache_dict_file.close()

    if(save_cache_checkpoint):
        cache_path = os.path.join(".", "cache_temp")
        os.makedirs(cache_path, exist_ok=True)
        cache_dict_name = os.path.join(cache_path, "cache_" + str(workload_idx)+ "_" + str(cache_max_size))
        cache_dict_file = open(cache_dict_name, "wb")
        pickle.dump(retriever.get_cache_dict(), cache_dict_file)
        cache_dict_file.close()

if(config_idx==0):
    # print("Running small-block")
    llm_eval_run(
        dataset=dataset, chunk_size = 256, k_retrieve=k_retrieve, index_type="flatL2", query_limit = num_query, skip = num_skip
    )
    # print("Running small-block done!")
    # print("-------------------------")
elif(config_idx==1):
    # print("Running large-block")
    llm_eval_run(
        dataset=dataset, chunk_size = 1024, k_retrieve=k_retrieve, index_type="flatL2", query_limit = num_query, skip = num_skip
    )
    # print("Running large-block done!")
    # print("-------------------------")
elif(config_idx==2):
    # print("Running ivf")
    llm_eval_run(
        dataset=dataset, chunk_size = 256, k_retrieve=k_retrieve, index_type="ivfFlat", query_limit = num_query, skip = num_skip
    )
    # print("Running ivf done!")
    # print("-------------------------")
elif(config_idx>=3):
    # print("Running two-level, small")
    llm_eval_run2(
        dataset=dataset, k_retrieve=k_retrieve, nprobe=nprobe, query_limit = num_query, skip = num_skip, config_idx = config_idx
    )
    # print("Running two-level done!")
    # print("-------------------------")


    dataset="nfcorpus",
    chunk_size=256,
    k_retrieve=1,
    index_type="flatL2",
    nprobe=1,
    query_limit=10,
    skip=0
