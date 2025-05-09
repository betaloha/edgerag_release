class RAG_Stats:
    aggregated_retrieve_time = 0
    aggregated_first_token_time = 0
    aggregated_generation_time = 0   # Generation time includes first token time
    aggregated_num_generated_tokens = 0
    aggregated_index_time = 0 # For indexing
    num_prompts = 0