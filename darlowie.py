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
#   - prefix JEXCLI - specific to Jira Export CLI version
#   - prefix QUERY  - specific to Query app
#   - prefix QUECLI - specific to Query CLI version
#   - prefix QUEWEB - specific to Query web version
#   - prefix DISCOV - specific to Discovery app
#   - prefix DISCLI - specific to Discovery CLI version
#   - prefix DISWEB - specific to Discovery web version

import logging


context = {

    # Global

    "GLOBALdataFolder" : "../testdata/",     # root data folder
    "GLOBALloggerLevel" : logging.WARN,      # WARN or INFO
    "GLOBALllm_Provider" : "ollama",
    "GLOBALllm_Embed" : "nomic-embed-text:latest",
    "GLOBALembedding_URL" : "http://localhost:11434/api/embeddings",
    "GLOBALllm_Version" : "gpt-oss:120b-cloud",
#    "GLOBALllm_Version" : "gemma4:latest",
#    "GLOBALllm_Version" : "gemma4:31b-cloud",
    "GLOBALllm_URL" : "http://localhost:11434/v1",
    "GLOBALrag_Datapath" : "chromadb",

    # Generator app settings

    "GENERAdocumentFolder" : "generatordocuments/",          # folder for source documents
    "GENERArag_activity_cutoff" : 0.3,       # cut off distance for activity table match
    "GENERArag_scenario_cutoff" : 0.35,      # cut off distance for scenario table match
    "GENERAad_FileName": "jobDescriptions/2025-10-02-0001.txt", # file name of original text
    "GENERAad_JSONName": "jobDescriptions/2025-10-02-0001.txt.json", # file name of JSON results
    "GENERAword_FileName": "jobDescriptions/2025-10-02-0001.txt.resume.docx", # file name of Word results

    # Generator CLI settings

    "GENCLIsession_key": "GENERATOR",   # session name for Generator CLI
    "GENCLIstatus_FileName": "status.GENERATOR.json",  # status file name for Generator CLI

    # Indexer app settings

    "INDEXEjira_url": "https://darlowie-security.atlassian.net",   # Jira Cloud API
    "INDEXEdocumentFolder" : "indexerdocuments/",   # folder for indexer source documents
    "INDEXEdataFolder" : "indexerdata/",            # indexer interim data folder
    "INDEXEbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index
    
    # Indexer CLI settings

    "IDXCLIsession_key": "INDEXER",   # session name for Indexer CLI
    "IDXCLIstatus_FileName": "status.INDEXER.json",  # status file name for Indexer CLI

    # Jira Export CLI settings

    "JEXCLIdocumentFolder" : "indexerdocuments/",    # jira export source document folder
    "JEXCLIdataFolder" : "jiraexport/",  # jira export interim data folder
    "JEXCLIsession_key": "JIRAEXPORT",   # session name for Jira Export CLI
    "JEXCLIstatus_FileName": "status.JIRAEXPORT.json",  # status file name for Jira Export CLI
    "JEXCLIjira_max_results" : 999,  # maximum number of Jira records to export

    # Query app settings

    "QUERYdataFolder" : "indexerdocuments/",          # Interim data folder
    "QUERYbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index

    # Query CLI settings

    "QUECLIsession_key": "QUERY",   # session name  for Query CLI
    "QUECLIstatus_FileName": "status.QUERY.json",  # status file name for Query CLI
    "QUECLIoutputCount": 50,     # default number of results in output

    # Discovery app settings

    "DISCOVdocumentFolder" : "discoverydocuments/",          # folder for source documents
    "DISCOVdataFolder" : "discoverydata/",          # Interim data folder
    "DISCOVbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index

    # Discovery CLI settings

    "DISCLIsession_key": "DISCOVERY",   # session name  for Discovery CLI
    "DISCLIstatus_FileName": "status.DISCOVERY.json",  # status file name for Discovery CLI
    "DISCLIoutputCount": 50     # default number of results in output

}