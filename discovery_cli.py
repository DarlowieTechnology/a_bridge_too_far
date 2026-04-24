import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4
from typing import List
import threading
import argparse
from pprint import pprint


from pydantic_ai.usage import RunUsage
from anyascii import anyascii

# local
import darlowie
from common import ConfigCollection
from discovery_workflow import DiscoveryWorkflow
from queryService import QueryService



def testRun(discoveryWorkflow : DiscoveryWorkflow) -> list[str]:
    """ 
    Test for discovery app
    
    Args:
        discoveryWorkflow (DiscoveryWorkflow) - workflow object
    Returns:
        None    
    """
    totalStart = time.time()

    fileList = discoveryWorkflow.formFileList()
    fileListEngineering = [
        "engineering-000.json",
        "engineering-001.json",
        "engineering-002.json",
        "engineering-003.json",
        "engineering-004.json",
        "engineering-005.json",
        "engineering-006.json",
        "engineering-007.json",
        "engineering-008.json",
        "engineering-009.json",
        "engineering-010.json",
        "engineering-011.json",
        "engineering-012.json",
        "engineering-013.json",
        "engineering-014.json",
        "engineering-015.json",
        "engineering-016.json",
        "engineering-017.json",
        "engineering-018.json",
        "engineering-019.json"
    ]

    fileListMedical = [
        "medresearch-000.txt",
        "medresearch-001.txt",
        "medresearch-002.txt",
        "medresearch-003.txt",
        "medresearch-004.txt",
        "medresearch-005.txt",
        "medresearch-006.txt",
        "medresearch-007.txt",
        "medresearch-008.txt",
        "medresearch-009.txt",
        "medresearch-010.txt",
        "medresearch-011.txt",
        "medresearch-012.txt",
        "medresearch-013.txt",
        "medresearch-014.txt",
        "medresearch-015.txt",
        "medresearch-016.txt",
        "medresearch-017.txt",
        "medresearch-018.txt",
        "medresearch-019.txt"
    ]

    fileListLLM = [
        "1904.10509v1.pdf",
        "1912.02292v1.pdf",
        "1912.06680v1.pdf",
        "2005.00341v1.pdf",
        "2005.14165v4.pdf",
        "2009.03393v1.pdf",
        "2102.12092v2.pdf",
        "2103.00020v1.pdf",
        "2107.03374v2.pdf",
        "2110.05448v1.pdf",
        "2112.10741v3.pdf",
        "2202.01344v1.pdf",
        "2212.04356v1.pdf",
        "2303.01469v2.pdf",
        "2303.08774v6.pdf",
        "2305.20050v1.pdf",
        "2312.09390v1.pdf",
        "2406.04093v1.pdf",
        "2410.21276v1.pdf",
        "2412.16720v1.pdf"
    ]

    fileListPenTest = [
        "AWS_Review.pdf",
        "CD_and_DevOps Review.pdf",
        "Database Review.pdf",
        "Firewall Review.pdf",
        "phpMyAdmin.pdf",
        "PHP_Code_Review.pdf",
        "Refinery-CMS.pdf",
        "WASPT_Report.pdf",
        "Web App and Ext Infrastructure Report.pdf",
        "Web App and Infrastructure and Mobile Report.pdf",
        "Wikimedia.pdf"
    ]

    fullFileList = fileListEngineering + fileListMedical + fileListLLM + fileListPenTest

#    fileList = [
#        "medresearch-000.txt"
#        "Refinery-CMS.pdf"
#        "2412.16720v1.pdf"
#        "AWS_Review.pdf"
#        "Database Review.pdf"
#        "2009.03393v1.pdf"
#        "1912.02292v1.pdf"
#    ]

    fileList = fileListMedical


    msg = f"Discovered {len(fileList)} files for processing."
    discoveryWorkflow.workerSnapshot(msg)

    if discoveryWorkflow.loadDocument:
        startTime = time.time()
        discoveryWorkflow.loadDocumentPhaseAllFiles(inputFileList = fileList)
        discoveryWorkflow.updateStats(topKey = "Load Documents", keyValList = [("Time", time.time() - startTime)])

    if discoveryWorkflow.parseChunks:
        startTime = time.time()
        discoveryWorkflow.parseChunksPhaseAllFiles(inputFileList = fileList)
        discoveryWorkflow.updateStats(topKey = "Chunking", keyValList = [("Time", time.time() - startTime)])

    if discoveryWorkflow.makeRawVector:
        startTime = time.time()
        accepted, rejected = discoveryWorkflow.makeRawVectorPhaseAllFiles(inputFileList = fileList)
        discoveryWorkflow.updateStats(topKey = "Vectorizing", keyValList = [("Time", time.time() - startTime), ("Chunks Accepted", accepted), ("Chunks Rejected", rejected)])

    if discoveryWorkflow.bm25Process:
        startTime = time.time()
        discoveryWorkflow.bm25ProcessPhaseAllFiles(inputFileList = fileList)
        discoveryWorkflow.updateStats(topKey = "BM25 Process", keyValList = [("Time", time.time() - startTime)])

    if discoveryWorkflow.matchChunks:
        startTime = time.time()
        queryService = QueryService()
        allQueryResults = discoveryWorkflow.matchChunksPhase(queryTexts = discoveryWorkflow.knownTopics, queryService = queryService)

        # output results files
        print(f"Output file name: {discoveryWorkflow.outputFileName}")
        with open(discoveryWorkflow.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(allQueryResults.model_dump_json(indent=2))


        msgList = discoveryWorkflow.outputRRFInfo(allQueryResults.rrfScores)
        print(json.dumps(msgList, indent = 4))
#        self.workerSnapshot(msgList)
        discoveryWorkflow.updateStats(topKey = "Matching", keyValList = [("Time", time.time() - startTime)])

    if discoveryWorkflow.clear:
        startTime = time.time()
        discoveryWorkflow.clearPhaseAllFiles(inputFileList = fileList)
        discoveryWorkflow.updateStats(topKey = "Clearing", keyValList = [("Time", time.time() - startTime)])


    discoveryWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart)])

    msg = f"{pprint(discoveryWorkflow.stats)}"
    discoveryWorkflow.workerSnapshot(msg)


def main():

    context = darlowie.context

    parser = argparse.ArgumentParser(description="Discovery CLI")
    parser.add_argument("--query", help="User query (\"medical notes\" or \"pipe engineering\")")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--count", help="Count of results in output")
    args = parser.parse_args()
    if args.query:
        userQuery = args.query
    else:
        print("Provide user query")
        return
    if args.output:
        outputFileName = args.output
    else:
        outputFileName = context["GLOBALdataFolder"] + context["DISCOVdocumentFolder"] + "DISCOVERY.results.json"
    if args.count:
        outputNumber = args.output
    else:
        outputNumber = 50

    # ------ configurable on command line
    #
    context["knownTopics"] = [userQuery]            # query - configurable on command line
    context["outputNumber"] = outputNumber          # number of items in result lists - configurable on command line
    context['outputFileName'] = outputFileName      # name of output file - configurable on command line

    # ------ other configuration parameter
    #
    # configuration of base class
    context["statusFileName"] = context["DISCLIstatus_FileName"]
    context["session_key"] = context["DISCLIsession_key"]
    context["ragDatapath"] = context["GLOBALdataFolder"] +  context["DISCOVdocumentFolder"] + context["GLOBALrag_Datapath"]

    # workflow actions
    context["loadDocument"] = True
    context["parseChunks"] = True
    context["makeRawVector"] = True
    context["bm25Process"] = True
    context["matchChunks"] = False
    context["verify"] = False
    context["returnResults"] = False
    context["clear"] = False

    # text extraction configuration
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    # other app-specific configuration
    context["fileExtensions"] = ["*.txt", "*.pdf", "*.json"]
    context["chunkSize"] = 256
    context["chunkOverlap"] = 48

    context["searchSemanticOriginal"] = True
    context["searchBM25sOriginal"] = True
    context["searchSemanticMulti"] = True
    context["searchBM25sMulti"] = True
    context["searchSemanticRewrite"] = True
    context["searchBM25sRewrite"] = True
    context["searchSemanticHyDE"] = True
    context["searchBM25sHyDE"] = True

    # retrieval configuration
    context["semanticRetrieveNumber"] = 1000        # maximum number of semantic items to retrieve
    context["semanticMaxCutItemDistance"] = 1.0     # distance cut-off for semantic matches
    context["bm25sRetrieveNumber"] = 1000           # maximum number of bm25s items to retrieve
    context["bm25sMinCutOffScore"] = 0.0            # bm25s score cut-off
    context["rrfCutOffValue"] = 0.00                # minimal RRF score to cut-off
    context["rrfOutlierZScoreThreshold"] = 3        # Z-score threshold for outliers

    configCollection = ConfigCollection(context)

    discoverWorkflow = DiscoveryWorkflow()
    discoverWorkflow.configure(configCollection)

    testRun(discoverWorkflow)

#    thread = threading.Thread( target=discoverWorkflow.threadWorker)
#    thread.start()
#    thread.join()


if __name__ == "__main__":
    main()
