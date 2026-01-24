#
# initial settings for all Darlowie applications (web apps and CLI)
# All applications should copy on start to avoid conflicts
# Naming rules for keys:
#   - prefix GLOBAL - common parameter like LLM name, URL for LLM host
#   - prefix GENERA - specific to Generator app
#   - prefix GENCLI - specific to Generator CLI version
#   - prefix GENWEB - specific to Generator web version
#   - prefix INDEXE - specific to Indexer app
#   - prefix IDXCLI - specific to Indexer CLI version
#   - prefix IDXWEB - specific to Indexer web version
#   - prefix QUERY  - specific to Query app
#   - prefix QUECLI - specific to Query CLI version
#   - prefix QUEWEB - specific to Query web version
#   - prefix DISCOV - specific to Discovery app
#   - prefix DISCLI - specific to Discovery CLI version
#   - prefix DISWEB - specific to Discovery web version


context = {

    # Global settings

    "GLOBALrag_datapath": "chromadb",    # Path to folder with vector database, used to create ChromaClient
    "GLOBALrag_embed_llm": "nomic-embed-text",   # Embedding LLM for RAG applications, used to create OllamaEmbeddingFunction
    "GLOBALrag_embed_url": "http://localhost:11434/api/embeddings", # URL for embedding API, used to create OllamaEmbeddingFunction
    "GLOBALrag_hnsw_space": "cosine",  # For Hierarchical Navigable Small World (HNSW) search algorithm choose cosine similarity metric
    "GLOBALllm_Provider": "Ollama",   # LLM service provider
    "GLOBALllm_Version": "gemini-3-flash-preview:latest",  # LLM name
    "GLOBALllm_base_url": "http://localhost:11434/v1", # OpenAI API endpoint

    # Generator app settings

    "GENERArag_activity_cutoff": 0.3,       # cut off distance for activity table match
    "GENERArag_scenario_cutoff": 0.35,      # cut off distance for scenario table match
    "GENERAad_FileName": "jobDescriptions/2025-10-02-0001.txt", # file name of original text
    "GENERAad_JSONName": "jobDescriptions/2025-10-02-0001.txt.json", # file name of JSON results
    "GENERAword_FileName": "jobDescriptions/2025-10-02-0001.txt.resume.docx", # file name of Word results

    # Generator CLI settings

    "GENCLIsession_key": "GENERATOR",   # session name for Generator CLI
    "GENCLIstatus_FileName": "status.GENERATOR.json",  # status file name for Generator CLI

    # Indexer app settings

    "INDEXEjira_url": "https://darlowie-security.atlassian.net",   # Jira Cloud API
    "INDEXEjira_max_results" : 1000,  # maxumum number of Jira records to export
    "INDEXEjira_export": False,  # Perform Jira export
    
    # Indexer CLI settings

    "IDXCLIsession_key": "INDEXER",   # session name for Indexer CLI
    "IDXCLIstatus_FileName": "status.INDEXER.json",  # status file name for Indexer CLI

    # Query CLI settings

    "QUECLIsession_key": "QUERY",   # session name  for Query CLI
    "QUECLIstatus_FileName": "status.QUERY.json",  # status file name for Query CLI

    # Discovery CLI settings

    "DISCLIsession_key": "DISCOVERY",   # session name  for Discovery CLI
    "DISCLIstatus_FileName": "status.DISCOVERY.json"  # status file name for Discovery CLI

}
