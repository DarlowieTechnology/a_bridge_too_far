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


def testRun(context : Dict[str, Any], logger : Logger, fileList : List[str]):
    """ 
    Test for indexer phases
    
    Args:
        indexerWorkflow (IndexerWorkflow) - workflow object
        logger (Logger) - app logger
        fileList(List[str]) = list of files to process
    
    """

    corpus : List[str] = []

    # read template description
    documentJSONName = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + "documents.json"
    with open(documentJSONName, "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    for fileName in fileList:

        context["inputFileName"] = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + fileName
        context["rawTextFromDoc"] = context["inputFileName"] + ".raw.txt"
        context["rawJSON"] = context["inputFileName"] + ".raw.json"
        context["finalJSON"] = context["inputFileName"] + ".json"
        context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)
        
        # raw text parsing support
        inputFileBaseName = context["inputFileBaseName"]

        context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
        context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
        context["extractPattern"] = dictDocuments[inputFileBaseName]["extract"]
        context["assignList"] = dictDocuments[inputFileBaseName]["assign"]

        configCollection = ConfigCollection(context)
        indexerWorkflow = IndexerWorkflow()
        indexerWorkflow.configure(configCollection)

        issueTemplate = ParserClassFactory.factory(context["issueTemplate"])
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
        IndexerWorkflow.bm25sProcessCorpusPhase(corpus=corpus, folderName = context["bm25IndexFolder"])


    msg = f"{pprint(indexerWorkflow.stats)}"
    indexerWorkflow.workerSnapshot(msg)



def main():

    context = darlowie.context

    context["statusFileName"] = context["IDXCLIstatus_FileName"]
    context["session_key"] = context["IDXCLIsession_key"]

    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logger = logging.getLogger(context["IDXCLIsession_key"])

    # test list - only process data sources from this list
    fileList = [
#        "Architecture Review - Threat Model Report.pdf",
#        "AWS_Review.pdf",
#        "CD_and_DevOps Review.pdf",
#        "Database Review.pdf",
#        "Firewall Review.pdf",
#        "phpMyAdmin.pdf",
#        "PHP_Code_Review.pdf",
#        "WASPT_Report.pdf",
#        "Web App and Ext Infrastructure Report.pdf",
#        "Wikimedia.pdf",
#        "Web App and Infrastructure and Mobile Report.pdf",
        "Refinery-CMS.pdf"
    ]

    if context["INDEXEjira_export"]:

        testRun(context = context, logger=logger, fileList = ["SCRUM"])

#        thread = threading.Thread( target=IndexerWorkflow.threadWorkerStatic, args=(context, fileList))
#        thread.start()
#        thread.join()

    else:

        context["bm25IndexFolder"] = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + context["INDEXEbm25IndexFolder"]
        context["bm25CorpusFileName"] = context["INDEXEbm25CorpusFileName"]

        # stages
        context["loadDocument"] = True
        context["rawTextFromDocument"] = True
        context["finalJSONfromRaw"] = True
        context["prepareBM25corpus"] = True
        context["completeBM25database"] = False
        context["vectorizeFinalJSON"] = False

        # text extraction from PDF
        context["stripWhiteSpace"] = True
        context["convertToLower"] = True
        context["convertToASCII"] = True
        context["singleSpaces"] = True

        testRun(context=context, logger=logger, fileList = fileList)

#            thread = threading.Thread( target=IndexerWorkflow.threadWorkerStatic, args=(context, fileList))
#            thread.start()
#            thread.join()


if __name__ == "__main__":
    main()
