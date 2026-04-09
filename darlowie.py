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


context = {

    # Global

    "GLOBALdataFolder" : "../testdata/",     # root data folder

    # Generator app settings

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
    "INDEXEjira_max_results" : 999,  # maximum number of Jira records to export
    "INDEXEjira_export": False,  # Perform Jira export
    "INDEXEdataFolder" : "indexerdocuments/",          # Interim data folder
    "INDEXEbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index
    "INDEXEbm25CorpusFileName" : "corpus.jsonl",    # name of corpus file
    
    # Indexer CLI settings

    "IDXCLIsession_key": "INDEXER",   # session name for Indexer CLI
    "IDXCLIstatus_FileName": "status.INDEXER.json",  # status file name for Indexer CLI

    # Jira Export CLI settings

    "JEXCLIdataFolder" : "indexerdocuments/jiraexport/",          # Interim data folder
    "JEXCLIsession_key": "JIRAEXPORT",   # session name for Jira Export CLI
    "JEXCLIstatus_FileName": "status.JIRAEXPORT.json",  # status file name for Jira Export CLI

    # Query app settings

    "QUERYdataFolder" : "indexerdocuments/",          # Interim data folder
    "QUERYbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index
    "QUERYbm25CorpusFileName" : "corpus.jsonl",    # name of corpus file


    # Query CLI settings

    "QUECLIsession_key": "QUERY",   # session name  for Query CLI
    "QUECLIstatus_FileName": "status.QUERY.json",  # status file name for Query CLI

    # Discovery app settings

    "DISCOVdocumentFolder" : "discoverydocuments/",          # folder for source documents
    "DISCOVdataFolder" : "discoverydata/",          # Interim data folder
    "DISCOVbm25IndexFolder" : "__combined.bm25/",   # folder for combined BM25 index
    "DISCOVbm25CorpusFileName" : "corpus.jsonl",    # name of corpus file

    # Discovery CLI settings

    "DISCLIsession_key": "DISCOVERY",   # session name  for Discovery CLI
    "DISCLIstatus_FileName": "status.DISCOVERY.json"  # status file name for Discovery CLI

}
