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
from common import GLOBALPROVIDER, LLMNAMES, CommonHelper, ConfigCollection, DebugUtils, OpenFile, RecordCollection
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
    parser.add_argument("--provider", help=f"LLM service provider, for full list pass \"--provider ?\"")
    parser.add_argument("--llm", help=f"LLM name, for full list pass \"---llm ?\"")
    parser.add_argument("--verbose", help=f"Verbosity, one of [{logging.INFO}, {logging.WARN}]")
    parser.add_argument("--input", help="File with reports to process, new line delimited")
    parser.add_argument("--load", action='store_const', const=True, help=f"Perform PDF load")
    parser.add_argument("--rawjson", action='store_const', const=True, help=f"Perform raw JSON extraction")
    parser.add_argument("--finaljson", action='store_const', const=True, help=f"Perform final JSON extraction")
    parser.add_argument("--bm25s", action='store_const', const=True, help=f"Create bm25s index")
    parser.add_argument("--vectorize", action='store_const', const=True, help=f"Perform vectorization")
    parser.add_argument("--clear", action='store_const', const=True, help=f"Perform temp files removal")
    args = parser.parse_args()

    if args.provider:
        if args.provider == '?':
            CommonHelper.displayProviderLLM(context)
            return
        if args.provider not in GLOBALPROVIDER:
            print(f"Unknown provider {args.provider}")
            return
        else:
            context['GLOBALllm_Provider'] = args.provider

    if args.llm:
        if args.llm == '?':
            CommonHelper.displayProviderLLM(context)
            return
        if args.llm not in LLMNAMES:
            print(f"Unknown LLM {args.llm}")
            return
        else:
            provider = context["GLOBALllm_Provider"]
            CommonHelper.setLLMName(provider, args.llm)

    if args.verbose:
        # can be any logging.XXXX values, so we don't check, see Python logging package for details
        context['GLOBALloggerLevel'] = int(args.verbose)

    # stages
    if args.load:
        context["loadDocument"] = True
    else:
        context["loadDocument"] = False

    if args.rawjson:
        context["rawTextFromDocument"] = True
    else:
        context["rawTextFromDocument"] = False

    if args.finaljson:
        context["finalJSONfromRaw"] = True
    else:
        context["finalJSONfromRaw"] = False

    if args.bm25s:
        context["prepareBM25corpus"] = True
    else:
        context["prepareBM25corpus"] = False

    if args.vectorize:
        context["vectorizeFinalJSON"] = True
    else:
        context["vectorizeFinalJSON"] = False

    if args.clear:
        context["clear"] = True
    else:
        context["clear"] = False

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

    # summary of command line
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")

    # text extraction from PDF
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    configCollection = ConfigCollection()
    configCollection.configure(context = context)

    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)

#    testRun(indexerWorkflow = indexerWorkflow, fileList = [fileList])

    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=([fileList]))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
