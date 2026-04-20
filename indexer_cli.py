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
from pprint import pprint

from typing import Any, List, Dict

from pydantic import BaseModel, Field

# local
import darlowie
from common import COLLECTION, ConfigCollection, DebugUtils, OpenFile, RecordCollection
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

    # test list - only process data sources from this list
    fileList = [
        "Architecture Review - Threat Model Report.pdf"
        ,
        "AWS_Review.pdf"
        ,
        "CD_and_DevOps Review.pdf"
        ,
        "Database Review.pdf"
        ,
        "Firewall Review.pdf"
        ,
        "phpMyAdmin.pdf"
        ,
        "PHP_Code_Review.pdf"
        ,
        "WASPT_Report.pdf"
         ,
        "Web App and Ext Infrastructure Report.pdf"
         ,
        "Wikimedia.pdf"
        ,
        "Web App and Infrastructure and Mobile Report.pdf"
         ,
        "Refinery-CMS.pdf"
    ]

    # stages
    context["loadDocument"] = False
    context["rawTextFromDocument"] = True
    context["finalJSONfromRaw"] = False
    context["prepareBM25corpus"] = False
    context["vectorizeFinalJSON"] = False
    context["clear"] = False

    # text extraction from PDF
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    # configuration of base class
    context["statusFileName"] = context["IDXCLIstatus_FileName"]
    context["session_key"] = context["IDXCLIsession_key"]

    configCollection = ConfigCollection(context)
    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)

#    testRun(indexerWorkflow = indexerWorkflow, fileList = [fileList])

    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=([fileList]))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
