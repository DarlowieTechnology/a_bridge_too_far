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
from common import GLOBALPROVIDER, LLMNAMES, CommonCLIArguments, CommonHelper, ConfigCollection, OpenFile, DebugUtils
from discovery_workflow import DiscoveryWorkflow
from queryService import QueryService
from resultsQueryClasses import CollectionChunkQueryResults


def testRun(discoveryWorkflow : DiscoveryWorkflow) :
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

    fileList = [
#        "medresearch-000.txt"
        "Refinery-CMS.pdf"
#        "2412.16720v1.pdf"
#        "AWS_Review.pdf"
#        "Database Review.pdf"
#        "2009.03393v1.pdf"
#        "1912.02292v1.pdf"
    ]

    
    fileList = discoveryWorkflow.source
#    fileList = fullFileList

    if discoveryWorkflow.loadDocument or discoveryWorkflow.parseChunks or discoveryWorkflow.makeRawVector or discoveryWorkflow.bm25Process or discoveryWorkflow.clear:
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

    if discoveryWorkflow.search:
        startTime = time.time()
        queryService = QueryService()

        collectionChunkQueryResults = discoveryWorkflow.matchChunksPhaseAllQueries(queryTexts = discoveryWorkflow.query, queryService = queryService)

        # output results files
        print(f"Output file name: {discoveryWorkflow.outputFileName}")
        with open(discoveryWorkflow.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(collectionChunkQueryResults.model_dump_json(indent=2))

        for queryResult in collectionChunkQueryResults.listAllQueryResults:
            msgList = discoveryWorkflow.outputRRFInfo(rrfScores = queryResult.rrfScores, onlyOutliers = True)
#        print(json.dumps(msgList, indent = 4))
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
    parser.add_argument("--provider", help=f"LLM service provider, for full list use \"--provider ?\"")
    parser.add_argument("--llm", help=f"LLM name, for full list use \"---llm ?\"")
    parser.add_argument("--source", help=f"Source file name")
    parser.add_argument("--sourcefiles", help=f"List of sources in text file, new line delimited")
    parser.add_argument("--query", help="User query string, for example \"Bell's palsy\"")
    parser.add_argument("--input", help="User queries in text file, new line delimited")
    parser.add_argument("--output", help=f"Output file with search results, default \"{context['DISCOVOutFile']}\"")
    parser.add_argument("--count", help=f"Count of results in output, default {context['DISCLIoutputCount']}")
    parser.add_argument("--verbose", help=f"Verbosity, one of [DEBUG, INFO, WARN, ERROR, CRITICAL]")
    parser.add_argument("--advanced", help=f"Advanced configuration JSON file")
    parser.add_argument("--showconfiguration", action='store_const', const=True, help="Show workflow configuration")
    parser.add_argument("--load", action='store_const', const=True, help=f"Load documents")
    parser.add_argument("--parsechunks", action='store_const', const=True, help=f"Parse chunks")
    parser.add_argument("--makerawvector", action='store_const', const=True, help=f"Create raw vector table")
    parser.add_argument("--bm25s", action='store_const', const=True, help=f"Create bm25s index")
    parser.add_argument("--search", action='store_const', const=True, help=f"Execute hybrid search")
    parser.add_argument("--clear", action='store_const', const=True, help=f"Remove temp files")

    args = parser.parse_args()

    # process advanced configuration first, named parameters below supersede advanced configuration
    if args.advanced:
        context = CommonCLIArguments.processAdvanced(args.advanced, context)
    if args.provider:
        context = CommonCLIArguments.processProvider(args.provider, context)
    if args.llm:
        context = CommonCLIArguments.processLLM(args.llm, context)

    if args.source:
        # merge without duplicates with list from advanced
        if "source" in context.keys():
            context["source"] = list(set(context["source"] + [args.source]))
        else:
            context["source"] = [args.source]

    if args.sourcefiles:
        res, errOrContent = OpenFile.open(filePath = args.sourcefiles, readContent = True)
        if not res:
            print(errOrContent)
            return
        fileList = errOrContent.split('\n')
        fileList = [x for x in fileList if x]   # remove empty strings
        if "source" in context.keys():
            # merge without duplicates with list from --advanced and --source
            context["source"] = list(set(context["source"] + fileList))
        else:
            context["source"] = fileList

    if ("source" not in context.keys()) and (args.load or args.parsechunks or args.makerawvector or args.bm25s):
        print("ERROR: Provide --source or --sourcefiles parameters")
        return

    if args.verbose:
        context.setdefault('logginglevel', CommonHelper.convertName2LoggingLevel(args.verbose))

    # phases
    if args.load:
        context["loadDocument"] = True
    else:
        context.setdefault("loadDocument", False)

    if args.parsechunks:
        context["parseChunks"] = True
    else:
        context.setdefault("parseChunks", False)

    if args.makerawvector:
        context["makeRawVector"] = True
    else:
        context.setdefault("makeRawVector", False)

    if args.bm25s:
        context["bm25Process"] = True
    else:
        context.setdefault("bm25Process", False)

    if args.search:
        context["search"] = True
    else:
        context.setdefault("search", False)

    if args.clear:
        context["clear"] = True
    else:
        context.setdefault("clear", False)

    if context["search"]:
        # combine --query, --input and --advanced values
        querySet = set()

        if args.input:
            res, errOrContent = OpenFile.open(filePath = args.input, readContent = True)
            if not res:
                print(errOrContent)
                return
            queryList = errOrContent.split('\n')
            queryList = [x for x in queryList if x]   # remove empty strings
            querySet.update(queryList)

        if args.query:
            querySet.add(args.query)

        if "query" in context.keys():
            # process --advanced, could be str or list[str]
            if type(context['query']) == str:
                querySet.add(context['query'])
            else:
                querySet.update(context['query'])

        context["query"] = list(querySet)
        if (not len(context["query"])):
            print("ERROR: Provide --query or --input parameters")
            return

    if (len(context["query"]) and not context["search"]):
        print("ERROR: Provide --search to perform search phase")
        return

    if args.output:
        context['outputFileName'] = args.output
    else:
        context.setdefault('outputFileName', context['DISCOVOutFile'])

    if args.count:
        context["outputNumber"] = args.output
    else:
        context.setdefault('outputNumber', context['DISCLIoutputCount'])

    if args.showconfiguration:
        showFlag = True
    else:
        showFlag = False

    # ------ configuration parameters default settings
    #

    # text extraction configuration
    context.setdefault("stripWhiteSpace", True)
    context.setdefault("convertToLower", True)
    context.setdefault("convertToASCII", True)
    context.setdefault("singleSpaces", True)

    # other app-specific configuration
    context.setdefault("fileExtensions", ["*.txt", "*.pdf", "*.json"])
    context.setdefault("chunkSize", 256)    
    context.setdefault("chunkOverlap", 48)    

    # components of hybrid search
    context.setdefault("searchSemanticOriginal", True)
    context.setdefault("searchBM25sOriginal", True)
    context.setdefault("searchSemanticMulti", True)
    context.setdefault("searchBM25sMulti", True)
    context.setdefault("searchSemanticRewrite", True)
    context.setdefault("searchBM25sRewrite", True)
    context.setdefault("searchSemanticHyDE", True)
    context.setdefault("searchBM25sHyDE", True)

    # retrieval configuration
    context.setdefault("semanticRetrieveNumber", 1000)        # maximum number of semantic items to retrieve
    context.setdefault("semanticMaxCutItemDistance", 1.0)     # distance cut-off for semantic matches
    context.setdefault("bm25sRetrieveNumber", 1000)           # maximum number of bm25s items to retrieve
    context.setdefault("bm25sMinCutOffScore", 0.0)            # bm25s score cut-off
    context.setdefault("rrfCutOffValue", 0.0)                 # minimal RRF score to cut-off
    context.setdefault("rrfOutlierZScoreThreshold", 15)       # Z-score threshold for outliers (typically 3)
    context.setdefault("rrfOutlierIQRCoefficient", 20.0)      # Interquartile Range (IQR) upper fence coefficient (typically 1.5)

    # output some info about command line arguments
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")

    configCollection = ConfigCollection()
    configCollection.configure(context = context)

    discoverWorkflow = DiscoveryWorkflow()
    discoverWorkflow.configure(configCollection)

    if showFlag:
        discoverWorkflow.showConfiguration()


#    testRun(discoverWorkflow)

    thread = threading.Thread( target=discoverWorkflow.threadWorker)
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
