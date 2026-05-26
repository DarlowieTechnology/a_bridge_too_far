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


GLOBALdataFolder = "../testdata/"

context = {

    # Global

    "GLOBALdataFolder" : "../testdata/",     # root data folder
    "GLOBALloggerLevel" : logging.INFO,      # CRITICAL, ERROR, WARN, INFO
    "GLOBALllm_Provider" : "lmstudio",
    "GLOBALrag_Datapath" : "chromadb",
    "GLOBALloggerSessionKey" : "APPLOG",

    # Generator app settings

    "GENERAdocumentFolder" : "generatordocuments/",          # folder for source documents
    "GENERArag_activity_cutoff" : 0.3,       # cut off distance for activity table match
    "GENERArag_scenario_cutoff" : 0.35,      # cut off distance for scenario table match
    "GENERAad_FileName": "jobDescriptions/2025-10-02-0001.txt", # file name of original text
    "GENERAad_JSONName": "jobDescriptions/2025-10-02-0001.txt.json", # file name of JSON results
    "GENERAword_FileName": "jobDescriptions/2025-10-02-0001.txt.resume.docx", # file name of Word results

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

    "QUECLIoutputCount": 50,     # default number of results in output

    # Discovery app settings

    "DISCOVdocumentFolder" : "discoverydocuments/",          # folder for source documents
    "DISCOVdataFolder" : "discoverydata/",          # Interim data folder
    "DISCOVbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index

    # Discovery CLI settings

    "DISCLIoutputCount": 50     # default number of results in output

}