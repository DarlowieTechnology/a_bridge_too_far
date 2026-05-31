#
# initial settings for all Darlowie applications (web apps and CLI)
# All applications should copy on start to avoid conflicts
# Naming rules for keys:
#   - prefix GLOBAL - common parameter like LLM name, URL for LLM host
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


GLOBALdataFolder = "../testdata/"

context = {

    # Global

    "GLOBALdataFolder" : "../testdata/",     # root data folder
    "logginglevel" : logging.INFO,      # CRITICAL, ERROR, WARN, INFO
    "GLOBALllm_Provider" : "lmstudio",
    "GLOBALloggerSessionKey" : "APPLOG",

    # Indexer app settings

    "INDEXEdocumentFolder" : GLOBALdataFolder + "indexerdocuments/",   # folder for indexer source documents
    "INDEXEdataFolder" : GLOBALdataFolder + "indexerdocuments/indexerdata/",  # indexer interim data folder
    "INDEXEbm25IndexFolder" : GLOBALdataFolder + "indexerdocuments/__combined.bm25/",   # folder for combined BM25 index
    "INDEXERAGFolder" : GLOBALdataFolder + "indexerdocuments/chromadb/",   # folder for RAG database
    "INDEXEOutFile" : GLOBALdataFolder + "indexerdocuments/indexerdata/QUERY.results.json", # name of search output file
    
    # Jira Export CLI settings

    "JEXCLIjira_url": "https://darlowie-security.atlassian.net",   # Jira Cloud API
    "JEXCLIdocumentFolder" : "indexerdocuments/",    # jira export source document folder
    "JEXCLIdataFolder" : "jiraexport/",  # jira export interim data folder
    "JEXCLIjira_max_results" : 999,  # maximum number of Jira records to export

    # Query CLI settings

    "QUECLIoutputCount": "50",     # default number of results in output

    # Discovery app settings

    "DISCOVdocumentFolder" : GLOBALdataFolder + "discoverydocuments/",          # folder for source documents
    "DISCOVdataFolder" : GLOBALdataFolder + "discoverydocuments/discoverydata/",          # Interim data folder
    "DISCOVbm25IndexFolder" : GLOBALdataFolder + "discoverydocuments/__combined.bm25/",   # folder for combined BM25 index
    "DISCOVRAGFolder" : GLOBALdataFolder + "discoverydocuments/chromadb/",   # folder for RAG database
    "DISCOVOutFile" : GLOBALdataFolder + "discoverydocuments/discoverydata/DISCOVERY.results.json", # name of search output file

    # Discovery CLI settings

    "DISCLIoutputCount": "50"     # default number of results in output

}