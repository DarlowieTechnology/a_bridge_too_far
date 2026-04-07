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
import darlowie
from common import COLLECTION, ConfigCollection, DebugUtils, OpenFile, RecordCollection
from indexer_workflow import IndexerWorkflow
from parserClasses import ParserClassFactory


def loadDocumentPhase(indexerWorkflow) :

    indexerWorkflow.loadDocumentPhase()

    inputFileBaseName = str(Path(indexerWorkflow.inputFileName).name)
    msg = f"Read input document {inputFileBaseName}"
    indexerWorkflow.workerSnapshot(msg)


def preprocessReportRawTextPhase(indexerWorkflow):

    result, fileContentOrError = OpenFile.open(filePath = indexerWorkflow.rawTextFromDoc, readContent = True)
    if not result:
        msg = f"preprocess: {fileContentOrError} - perform 'loadDocument' action first"
        print(msg)
        return
    else:
        textCombined = fileContentOrError

    dictIssues = indexerWorkflow.preprocessReportRawTextPhase(textCombined)

    with open(indexerWorkflow.rawJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
        jsonOut.writelines(json.dumps(dictIssues, indent=2))

    rawJSONBaseName = str(Path(indexerWorkflow.rawJSON).name)
    msg = f"Preprocessed raw text {rawJSONBaseName}. Found {len(dictIssues)} potential issues."
    indexerWorkflow.workerSnapshot(msg)


def bm25preparePhase(indexerWorkflow : IndexerWorkflow, issueTemplate : BaseModel, corpus : list[str]):
    """
    Test for bm25s accumulation of corpus - called for each document
    
    :param indexerWorkflow: indexer workflow object
    :type indexerWorkflow: IndexerWorkflow
    :param issueTemplate: issue template for the document
    :type issueTemplate: BaseModel
    :param corpus: global corpus for bm25s
    :type corpus: list[str]
    """

    result, fileContentOrError = OpenFile.open(filePath = indexerWorkflow.finalJSON, readContent = True)
    if not result:
        msg = f"bm25prepare: {fileContentOrError} - perform 'preprocess' action first"
        print(msg)
        return
    else:
        recordCollection = RecordCollection.model_validate_json(fileContentOrError)

    indexerWorkflow.bm25sAddReportToCorpusPhase(corpus, recordCollection, issueTemplate)

    msg = f"Added {len(recordCollection.finding_dict)} records to global bm25s corpus."
    indexerWorkflow.workerSnapshot(msg)


def bm25complete(indexerWorkflow : IndexerWorkflow, corpus : list[str]):
    """
    Complete creation of BM25 database
    
    :param indexerWorkflow: indexer workflow object
    :type indexerWorkflow: IndexerWorkflow
    :param corpus: global corpus for bm25
    :type corpus: list[str]
    """
    folderName = indexerWorkflow.bm25IndexFolder
    indexerWorkflow.bm25sProcessCorpusPhase(corpus=corpus, folderName = folderName)
    msg = f"Indexed {len(corpus)} records for bm25 in folder {folderName}."
    indexerWorkflow.workerSnapshot(msg)


def parseIssuesPhase(indexerWorkflow : IndexerWorkflow, issueTemplate : BaseModel) :

    result, fileContentOrError = OpenFile.open(filePath = indexerWorkflow.rawJSON, readContent = True)
    if not result:
        msg = f"parseIssues: {fileContentOrError} - perform 'loadDocument' action first"
        print(msg)
        return
    else:
        dictIssues = json.loads(fileContentOrError)

    recordCollection = indexerWorkflow.parseAllIssuesPhase(dictIssues, issueTemplate)

    indexerWorkflow.writeFinalJSON(recordCollection)

    msg = f"Fetched {recordCollection.objectCount()} Wrote final JSON {str(Path(indexerWorkflow.finalJSON).name)}."
    indexerWorkflow.workerSnapshot(msg)


def vectorizeFinalJSONPhase(indexerWorkflow : IndexerWorkflow, issueTemplate : BaseModel):

    result, fileContentOrError = OpenFile.open(filePath = indexerWorkflow.finalJSON, readContent = True)
    if not result:
        msg = f"vectorizeFinalJSONPhase: {fileContentOrError} - perform 'parseIssues' action first"
        indexerWorkflow.workerSnapshot(msg)
        return
    else:
        recordCollection = RecordCollection.model_validate_json(fileContentOrError)

    accepted, rejected = indexerWorkflow.vectorizeFinalJSONPhase(recordCollection, issueTemplate)
    msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
    indexerWorkflow.workerSnapshot(msg)


def jiraExportPhase(indexerWorkflow, issueTemplate) -> bool:
    exportedIssues = indexerWorkflow.jiraExportPhase(issueTemplate)
    msg = f"Exported {exportedIssues} Jira issues."
    indexerWorkflow.workerSnapshot(msg)


def testRun(indexerWorkflow : IndexerWorkflow, logger : Logger, issueTemplate: BaseModel, corpus : list[str]) -> list[str]:
    """ 
    Test for indexer stages, called for each report document
    
    Args:
        indexerWorkflow (IndexerWorkflow) - workflow object
        logger (Logger) - app logger
        issueTemplate (BaseModel) - template for issue for the document
        corpus (list[str]) - global corpus for bm25s
    Returns:
        updated corpus
    
    """

    if indexerWorkflow.INDEXEjira_export :
        msg = f"Processing Jira database"
    else:
        msg = f"Processing data source {indexerWorkflow.inputFileName}"
    indexerWorkflow.workerSnapshot(msg)

    if indexerWorkflow.INDEXEjira_export :
        jiraExportPhase(indexerWorkflow, issueTemplate)
        if indexerWorkflow.vectorizeFinalJSON :
            vectorizeFinalJSONPhase(indexerWorkflow, issueTemplate)
    else:
        if indexerWorkflow.loadDocument :
            loadDocumentPhase(indexerWorkflow)
        if indexerWorkflow.rawTextFromDocument :
            preprocessReportRawTextPhase(indexerWorkflow)
        if indexerWorkflow.finalJSONfromRaw :
            parseIssuesPhase(indexerWorkflow, issueTemplate)
        if indexerWorkflow.prepareBM25corpus :
            bm25preparePhase(indexerWorkflow, issueTemplate, corpus)
        if indexerWorkflow.vectorizeFinalJSON :
            vectorizeFinalJSONPhase(indexerWorkflow, issueTemplate)
        if indexerWorkflow.completeBM25database :
            bm25complete(indexerWorkflow, corpus)

    return corpus


def main():

    context = darlowie.context

    context['status'] = []
    context["issuePattern"] = None
    context["issueTemplate"] = None
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

    # read template description
    documentJSONName = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + "documents.json"
    with open(documentJSONName, "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    if context["INDEXEjira_export"]:
        context["inputFileName"] = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + "SCRUM"
        context["rawTextFromDoc"] = context["inputFileName"] + ".raw.txt"
        context["rawJSON"] = context["inputFileName"] + ".raw.json"
        context["finalJSON"] = context["inputFileName"] + ".json"
        context["inputFileBaseName"] = "jira:SCRUM"
        inputFileBaseName = context["inputFileBaseName"]
        context["issueTemplate"] = "JiraIssueRAG"

        issueTemplate = ParserClassFactory.factory(context["issueTemplate"])

        configCollection = ConfigCollection(context)

        indexerWorkflow = IndexerWorkflow()
        indexerWorkflow.configure(configCollection)

        testRun(indexerWorkflow=indexerWorkflow, logger=logger, issueTemplate=issueTemplate)

#        thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus]))
#        thread.start()
#        thread.join()


    else:
        # global corpus for bm25s = updated for every file, completed once for all files
        corpus = []

        for fileName in fileList:
            context["inputFileName"] = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + fileName
            context["rawTextFromDoc"] = context["inputFileName"] + ".raw.txt"
            context["rawJSON"] = context["inputFileName"] + ".raw.json"
            context["finalJSON"] = context["inputFileName"] + ".json"
            
            context["bm25IndexFolder"] = context["GLOBALdataFolder"] + context["INDEXEdataFolder"] + context["INDEXEbm25IndexFolder"]
            context["bm25CorpusFileName"] = context["INDEXEbm25CorpusFileName"]
            context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)
            inputFileBaseName = context["inputFileBaseName"]

            # stages
            context["loadDocument"] = True
            context["rawTextFromDocument"] = True
            context["finalJSONfromRaw"] = True
            context["prepareBM25corpus"] = True
            context["completeBM25database"] = True
            context["vectorizeFinalJSON"] = True

            # text extraction from PDF
            context["stripWhiteSpace"] = True
            context["convertToLower"] = True
            context["convertToASCII"] = True
            context["singleSpaces"] = True

            # raw text parsing support
            context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
            context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
            context["extractPattern"] = dictDocuments[inputFileBaseName]["extract"]
            context["assignList"] = dictDocuments[inputFileBaseName]["assign"]

            issueTemplate = ParserClassFactory.factory(context["issueTemplate"])

            configCollection = ConfigCollection(context)
            indexerWorkflow = IndexerWorkflow()
            indexerWorkflow.configure(configCollection)

            corpus = testRun(indexerWorkflow=indexerWorkflow, logger=logger, issueTemplate=issueTemplate, corpus=corpus)

#            thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus]))
#            thread.start()
#            thread.join()

        msg = f"Processing completed."
        indexerWorkflow.workerSnapshot(msg)


if __name__ == "__main__":
    main()
