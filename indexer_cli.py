#
# Indexer CLI app
#

import sys
import logging
from logging import Logger
import threading
import json
import re
from pathlib import Path

from typing import Any
from typing import List

from jira import JIRA

from pydantic import BaseModel, Field

# local
from common import COLLECTION, ConfigSingleton, DebugUtils, OpenFile, RecordCollection
from indexer_workflow import IndexerWorkflow
from parserClasses import ParserClassFactory


def loadPDF(context, indexerWorkflow) :
    textCombined = indexerWorkflow.loadPDF(context['inputFileName'])
    with open(context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
        rawOut.write(textCombined)
    inputFileBaseName = str(Path(context['inputFileName']).name)
    msg = f"Read input document {inputFileBaseName}"
    indexerWorkflow.workerSnapshot(msg)


def preprocess(context, indexerWorkflow):
    with open(context['rawtextfromPDF'], "r", encoding='utf8', errors='ignore') as txtIn:
        textCombined = txtIn.read()

    dictIssues = indexerWorkflow.preprocessReportRawText(textCombined)
    with open(context['rawJSON'], "w", encoding='utf8', errors='ignore') as jsonOut:
        jsonOut.writelines(json.dumps(dictIssues, indent=2))

    rawJSONBaseName = str(Path(context['rawJSON']).name)
    msg = f"Preprocessed raw text {rawJSONBaseName}. Found {len(dictIssues)} potential issues."
    indexerWorkflow.workerSnapshot(msg)


def bm25prepare(context : dict, indexerWorkflow : IndexerWorkflow, issueTemplate : BaseModel, corpus : list[str]):
    """
    Test for bm25s accumulation of corpus - called for each document
    
    :param context: context for execution
    :type context: dict
    :param indexerWorkflow: indexer workflow object
    :type indexerWorkflow: IndexerWorkflow
    :param issueTemplate: issue template for the document
    :type issueTemplate: BaseModel
    :param corpus: global corpus for bm25s
    :type corpus: list[str]
    """
    with open(context['finalJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
        jsonStr = json.load(jsonIn)
        recordCollection = RecordCollection.model_validate(jsonStr)
    indexerWorkflow.bm25sAddReportToCorpus(corpus, recordCollection, issueTemplate)
    msg = f"Added {len(recordCollection.finding_dict)} records to global bm25s corpus."
    indexerWorkflow.workerSnapshot(msg)


def bm25complete(context : dict, indexerWorkflow : IndexerWorkflow, corpus : list[str]):
    """
    Complete creation of BM25 database
    
    :param context: context for execution
    :type context: dict
    :param indexerWorkflow: indexer workflow object
    :type indexerWorkflow: IndexerWorkflow
    :param corpus: global corpus for bm25
    :type corpus: list[str]
    """
    folderName = context["bm25IndexFolder"]
    indexerWorkflow.bm25sProcessCorpus(corpus=corpus, folderName = folderName)
    msg = f"Indexed {len(corpus)} records for bm25 in folder {folderName}."
    indexerWorkflow.workerSnapshot(msg)


def parseIssues(context, indexerWorkflow, issueTemplate):
    with open(context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
        dictIssues = json.load(jsonIn)
    recordCollection = indexerWorkflow.parseAllIssues(context['inputFileName'], dictIssues, issueTemplate)

    indexerWorkflow.writeFinalJSON(recordCollection)

    msg = f"Fetched {recordCollection.objectCount()} Wrote final JSON {str(Path(context['finalJSON']).name)}."
    indexerWorkflow.workerSnapshot(msg)


def vectorize(context, indexerWorkflow, issueTemplate):
    with open(context["finalJSON"], "r", encoding='utf8', errors='ignore') as jsonIn:
        jsonStr = json.load(jsonIn)
        recordCollection = RecordCollection.model_validate(jsonStr)

    accepted, rejected = indexerWorkflow.vectorize(recordCollection, issueTemplate)
    msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
    indexerWorkflow.workerSnapshot(msg)


def jiraExport(indexerWorkflow, issueTemplate) -> bool:
    exportedIssues = indexerWorkflow.jiraExport(issueTemplate)
    msg = f"Exported {exportedIssues} Jira issues."
    indexerWorkflow.workerSnapshot(msg)

def testLock(context, logger) -> bool : 
    boolResult, sessionInfoOrError = OpenFile.open(context["statusFileName"], True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                logger.info("Process: Removing completed session file")
            else:    
                logger.info("Process: Existing async processing found - exiting")
                return False
        except:
            logger.info("Process: Removing corrupt session file")
    return True


def testRun(context : dict, indexerWorkflow : IndexerWorkflow, logger : Logger, issueTemplate: BaseModel, corpus : list[str]) -> list[str]:
    """ 
    Test for indexer stages, called for each report document
    
    Args:
        context (dict) - all information for test run
        indexerWorkflow (IndexerWorkflow) - workflow object
        logger (Logger) - app logger
        issueTemplate (BaseModel) - template for issue for the document
        corpus (list[str]) - global corpus for bm25s
    Returns:
        updated corpus
    
    """

#    if not testLock(context, logger) : 
#        return

    context["stage"] = "starting"
    if context['JiraExport']:
        msg = f"Processing Jira database"
    else:
        msg = f"Processing data source {context['inputFileName']}"
    indexerWorkflow.workerSnapshot(msg)

    if context["JiraExport"]:
        jiraExport(indexerWorkflow, issueTemplate)
        vectorize(context, indexerWorkflow, issueTemplate)
    else:
        if "loadDocument" in context and context["loadDocument"]:
            loadPDF(context, indexerWorkflow)
        if "rawTextFromDocument" in context and context["rawTextFromDocument"]:
            preprocess(context, indexerWorkflow)
        if "finalJSONfromRaw" in context and context["finalJSONfromRaw"]:
            parseIssues(context, indexerWorkflow, issueTemplate)
        if "prepareBM25corpus" in context and context["prepareBM25corpus"]:
            bm25prepare(context, indexerWorkflow, issueTemplate, corpus)
        if "vectorizeFinalJSON" in context and context["vectorizeFinalJSON"]:
            vectorize(context, indexerWorkflow, issueTemplate)
        if "completeBM25database" in context and context["completeBM25database"]:
            bm25complete(context, indexerWorkflow, corpus)

    return corpus



def main():

    context = {}
    context["session_key"] = "INDEXER"
    context["statusFileName"] = "status.INDEXER.json"
    context["llmProvider"] = "Ollama"
#    context["llmVersion"] = "gpt-oss:120b-cloud"
    context["llmVersion"] = "gemini-3-flash-preview:latest"
    context["llmBaseUrl"] = "http://localhost:11434/v1"

    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []
    context["issuePattern"] = None
    context["issueTemplate"] = None
    context["JiraExport"] = False
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])


    # test list - only process data sources from this list
    fileList = [
#        "Architecture Review - Threat Model Report.pdf",
#        "AWS_Review.pdf",
#        "CD_and_DevOps Review.pdf",
#        "Database Review.pdf",
#        "Firewall Review.pdf",
#        "phpMyAdmin.pdf",
#        "PHP_Code_Review.pdf",
#        "Refinery-CMS.pdf",
#        "WASPT_Report.pdf",
#        "Web App and Ext Infrastructure Report.pdf",
#        "Wikimedia.pdf",
        "Web App and Infrastructure and Mobile Report.pdf"
    ]

    # read template description
    with open("webapp/indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    if context["JiraExport"]:
        context["inputFileName"] = "SCRUM"
        context["finalJSON"] = "webapp/indexer/input/SCRUM.json"
        context["inputFileBaseName"] = "jira:SCRUM"
        inputFileBaseName = context["inputFileBaseName"]
        context["issueTemplate"] = "JiraIssueRAG"
        issueTemplate = ParserClassFactory.factory(context["issueTemplate"])
        indexerWorkflow = IndexerWorkflow(context, logger)
        testRun(context=context, indexerWorkflow=indexerWorkflow, logger=logger, issueTemplate=issueTemplate)

#        thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus]))
#        thread.start()
#        thread.join()


    else:
        # global corpus for bm25s = updated for every file, completed once for all files
        corpus = []

        for fileName in fileList:
            context["inputFileName"] = "webapp/indexer/input/" + fileName
            context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
            context["rawJSON"] = context["inputFileName"] + ".raw.json"
            
            # folder for combined bm25s index - the same for all documents
            context["bm25IndexFolder"] = "webapp/indexer/input/combined.bm25s"

            context["finalJSON"] = context["inputFileName"] + ".json"
            context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)
            inputFileBaseName = context["inputFileBaseName"]

            # text extraction from PDF
            context["loadDocument"] = False
            context["stripWhiteSpace"] = True
            context["convertToLower"] = True
            context["convertToASCII"] = True
            context["singleSpaces"] = True

            # preprocess text
            context["rawTextFromDocument"] = False

            # create final JSON
            context["finalJSONfromRaw"] = True

            # prepare BM25s corpus
            context["prepareBM25corpus"] = False

            # complete BM25 database
            context["completeBM25database"] = False

            # vectorize final JSON
            context["vectorizeFinalJSON"] = False

            # raw text parsing support
            context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
            context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
            context["extractPattern"] = dictDocuments[inputFileBaseName]["extract"]
            context["assignList"] = dictDocuments[inputFileBaseName]["assign"]

            issueTemplate = ParserClassFactory.factory(context["issueTemplate"])
            indexerWorkflow = IndexerWorkflow(context, logger)
            corpus = testRun(context=context, indexerWorkflow=indexerWorkflow, logger=logger, issueTemplate=issueTemplate, corpus=corpus)

#            thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus]))
#            thread.start()
#            thread.join()

        context["stage"] = "completed"
        msg = f"Processing completed."
        indexerWorkflow.workerSnapshot(msg)


if __name__ == "__main__":
    main()
