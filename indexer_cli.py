#
# Indexer CLI app
#

import sys
import logging
from logging import Logger
import threading
import json
import re
import time
from pathlib import Path
import argparse
from pprint import pprint

from typing import Any, List, Dict

from pydantic import BaseModel, Field

# local
import darlowie
from common import GLOBALPROVIDER, LLMNAMES, CommonCLIArguments, CommonHelper, ConfigCollection, DebugUtils, OpenFile, RecordCollection
from indexer_workflow import IndexerWorkflow
from parserClasses import ParserClassFactory


def testRun(indexerWorkflow : IndexerWorkflow, fileList : List[List[str]]):
    """ 
    Test for indexer phases
    
    Args:
        indexerWorkflow (IndexerWorkflow) - workflow
        fileList(List[str]) = list of files to process
    
    """

    totalStart = time.time()

    if indexerWorkflow.loadDocument:
        indexerWorkflow.loadDocumentPhaseAllFiles(inputFileList = fileList[0])
    if indexerWorkflow.rawTextFromDocument :
        indexerWorkflow.rawTextFromDocumentPhaseAllFiles(inputFileList = fileList[0])
    if indexerWorkflow.finalJSONfromRaw :
        indexerWorkflow.finalJSONfromRawPhaseAllFiles(inputFileList = fileList[0])
    if indexerWorkflow.prepareBM25corpus :
        indexerWorkflow.prepareBM25corpusPhaseAllFiles(inputFileList = fileList[0])
    if indexerWorkflow.vectorizeFinalJSON :
        indexerWorkflow.vectorizeFinalJSONPhaseAllFiles(inputFileList = fileList[0])
    if indexerWorkflow.clear :
        indexerWorkflow.clearAllFiles()

    indexerWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart)])

    pprint(indexerWorkflow.stats)


def main():

    context = darlowie.context

    parser = argparse.ArgumentParser(prog = "indexer_cli.py", description="Indexer CLI")
    parser.add_argument("--provider", help=f"LLM service provider, for full list: \"--provider ?\"")
    parser.add_argument("--llm", help=f"LLM name, for full list:  \"---llm ?\"")
    parser.add_argument("--verbose", help=f"Verbosity, one of [DEBUG, INFO, WARN, ERROR, CRITICAL]")
    parser.add_argument("--advanced", help=f"Advanced configuration JSON file")
    parser.add_argument("--input", help="File with reports to process, new line delimited")
    parser.add_argument("--showconfiguration", action='store_const', const=True, help="Show workflow configuration")
    parser.add_argument("--load", action='store_const', const=True, help=f"Perform PDF load")
    parser.add_argument("--rawjson", action='store_const', const=True, help=f"Perform raw JSON extraction")
    parser.add_argument("--finaljson", action='store_const', const=True, help=f"Perform final JSON extraction")
    parser.add_argument("--bm25s", action='store_const', const=True, help=f"Create bm25s index")
    parser.add_argument("--vectorize", action='store_const', const=True, help=f"Perform vectorization")
    parser.add_argument("--clear", action='store_const', const=True, help=f"Perform temp files removal")

    args = parser.parse_args()

    # process advanced configuration first, named parameters below supersede advanced configuration
    if args.advanced:
        context = CommonCLIArguments.processAdvanced(args.advanced, context)
    if args.provider:
        context = CommonCLIArguments.processProvider(args.provider, context)
    if args.llm:
        context = CommonCLIArguments.processLLM(args.llm, context)

    if args.verbose:
        context.setdefault('logginglevel', CommonHelper.convertName2LoggingLevel(args.verbose))

    # phases
    if args.load:
        context["loadDocument"] = True
    else:
        context.setdefault("loadDocument", False)

    if args.rawjson:
        context["rawTextFromDocument"] = True
    else:
        context.setdefault("rawTextFromDocument", False)

    if args.finaljson:
        context["finalJSONfromRaw"] = True
    else:
        context.setdefault("finalJSONfromRaw", False)

    if args.bm25s:
        context["prepareBM25corpus"] = True
    else:
        context.setdefault("prepareBM25corpus", False)

    if args.vectorize:
        context["vectorizeFinalJSON"] = True
    else:
        context.setdefault("vectorizeFinalJSON", False)

    if args.clear:
        context["clear"] = True
    else:
        context.setdefault("clear", False)

    if context["loadDocument"] or context["rawTextFromDocument"] or context["finalJSONfromRaw"] or context["prepareBM25corpus"] or context["vectorizeFinalJSON"]:
        # enforce file list argument for all stages that require input files
        if not args.input:
            print("Provide list of reports to process")
            return

    if args.input:
        res, errOrContent = OpenFile.open(filePath = args.input, readContent = True)
        if not res:
            print(errOrContent)
            return
        fileList = errOrContent.split('\n')
        fileList = [x for x in fileList if x]   # remove empty strings
    else:
        fileList = []

    if args.showconfiguration:
        showFlag = True
    else:
        showFlag = False

    # text extraction from PDF
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    # output some info about command line arguments
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")


    configCollection = ConfigCollection()
    configCollection.configure(context = context)

    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)

    if showFlag:
        indexerWorkflow.showConfiguration()

#    testRun(indexerWorkflow = indexerWorkflow, fileList = [fileList])

    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=([fileList]))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
