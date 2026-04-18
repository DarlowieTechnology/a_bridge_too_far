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

from jira import JIRA

from pydantic import BaseModel, Field

# local
import darlowie
from common import COLLECTION, ConfigCollection, DebugUtils, OpenFile, RecordCollection
from indexer_workflow import IndexerWorkflow
from parserClasses import ParserClassFactory


def testRun(indexerWorkflow : IndexerWorkflow, fileList : List[str]):
    """ 
    Test for indexer phases
    
    Args:
        indexerWorkflow (IndexerWorkflow) - workflow object
        fileList(List[str]) = list of files to process
    
    """

    # bm25s index is common for all source documents
    corpus : List[str] = []

    if indexerWorkflow.loadDocument:
        startTime = time.time()
        indexerWorkflow.loadDocumentPhaseAllFiles(inputFileList = fileList)
        indexerWorkflow.updateStats(topKey = "Load Documents", keyValList = [("Time", time.time() - startTime)])

    pprint(indexerWorkflow.stats)
    return


    for fileName in fileList:

        context["inputFileBaseName"] = fileName
        context["interimFolder"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"]
        context["inputFileName"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + fileName
        context["rawTextFromDoc"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.txt"
        context["rawJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.json"
        context["finalJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".json"
        
        configCollection = ConfigCollection(context)
        indexerWorkflow = IndexerWorkflow()
        indexerWorkflow.configure(configCollection)

        issueTemplate = ParserClassFactory.factory(indexerWorkflow.issueTemplateName)
        if indexerWorkflow.loadDocument :
            indexerWorkflow.loadDocumentPhase()
        if indexerWorkflow.rawTextFromDocument :
            indexerWorkflow.rawTextFromDocumentPhase()
        if indexerWorkflow.finalJSONfromRaw :
            indexerWorkflow.finalJSONfromRawPhase(issueTemplate = issueTemplate)
        if indexerWorkflow.prepareBM25corpus :
            corpus = indexerWorkflow.prepareBM25corpusPhase(issueTemplate, corpus)
        if indexerWorkflow.vectorizeFinalJSON :
            indexerWorkflow.vectorizeFinalJSONPhase(issueTemplate)

    if indexerWorkflow.completeBM25database :
        folderName = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEbm25IndexFolder"]
        IndexerWorkflow.bm25sProcessCorpusPhase(corpus=corpus, folderName = folderName)

    pprint(indexerWorkflow.stats)


def main():

    context = darlowie.context

    # test list - only process data sources from this list
    fileList = [
        "Architecture Review - Threat Model Report.pdf",
        "AWS_Review.pdf",
        "CD_and_DevOps Review.pdf",
        "Database Review.pdf",
        "Firewall Review.pdf",
        "phpMyAdmin.pdf",
        "PHP_Code_Review.pdf",
        "WASPT_Report.pdf",
        "Web App and Ext Infrastructure Report.pdf",
        "Wikimedia.pdf",
        "Web App and Infrastructure and Mobile Report.pdf",
        "Refinery-CMS.pdf"
    ]

    # stages
    context["loadDocument"] = True
    context["rawTextFromDocument"] = False
    context["finalJSONfromRaw"] = False
    context["prepareBM25corpus"] = False
    context["completeBM25database"] = False
    context["vectorizeFinalJSON"] = False

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


    testRun(context=context, fileList = fileList)

#    thread = threading.Thread( target=IndexerWorkflow.threadWorkerStatic, args=(context, fileList))
#    thread.start()
#    thread.join()


if __name__ == "__main__":
    main()
