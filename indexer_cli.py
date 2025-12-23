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
from common import ConfigSingleton, DebugUtils, OpenFile, RecordCollection
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


def bm25prepare(context, indexerWorkflow, issueTemplate):
    with open(context['finalJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
        jsonStr = json.load(jsonIn)
        recordCollection = RecordCollection.model_validate(jsonStr)
    indexerWorkflow.bm25sProcessIssueText(recordCollection, issueTemplate)
    msg = f"Stored bm25s index in {context["bm25sJSON"]}."
    indexerWorkflow.workerSnapshot(msg)


def parseIssues(context, indexerWorkflow, issueTemplate):
    with open(context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
        dictIssues = json.load(jsonIn)
    recordCollection = indexerWorkflow.parseAllIssues(context['inputFileName'], dictIssues, issueTemplate)

    with open(context["finalJSON"], "w", encoding='utf8', errors='ignore') as jsonOut:
        jsonOut.writelines('{\n"finding_dict": {\n')
        idx = 0
        for key in recordCollection.finding_dict:
            jsonOut.writelines(f'"{key}" : ')
            jsonOut.writelines(recordCollection.finding_dict[key].model_dump_json(indent=2))
            idx += 1
            if (idx < len(recordCollection.finding_dict)):
                jsonOut.writelines(',\n')
        jsonOut.writelines('}\n}\n')

    finalJSONBaseName = str(Path(indexerWorkflow._context['finalJSON']).name)
    msg = f"Fetched {recordCollection.objectCount()} Wrote final JSON {finalJSONBaseName}."
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


def testRun(context : dict, indexerWorkflow : IndexerWorkflow, logger : Logger, issueTemplate: BaseModel) :
    """ 
    Test for indexer stages 
    
    Args:
        context (dict) - all information for test run
        issueTemplate (BaseModel) - template for issue
    Returns:
        None
    
    """

#    if not testLock(context, logger) : 
#        return

    context["stage"] = "starting"
    msg = f"Processing data source {context['inputFileName']}"
    indexerWorkflow.workerSnapshot(msg)

    if context["JiraExport"]:
        jiraExport(indexerWorkflow, issueTemplate)
        vectorize(context, indexerWorkflow, issueTemplate)
    else:
#        loadPDF(context, indexerWorkflow)
#        preprocess(context, indexerWorkflow)
#        parseIssues(context, indexerWorkflow, issueTemplate)
        bm25prepare(context, indexerWorkflow, issueTemplate)
        vectorize(context, indexerWorkflow, issueTemplate)

    context["stage"] = "completed"
    msg = f"Processing completed."
    indexerWorkflow.workerSnapshot(msg)


def main():

    context = {}
    context["session_key"] = "INDEXER"
    context["statusFileName"] = "status.INDEXER.json"
    context["llmProvider"] = "Ollama"
    context["llmOllamaVersion"] = "llama3.1:latest"
    context["llmBaseUrl"] = "http://localhost:11434/v1"
#    context["llmProvider"] = "Gemini"
#    context["llmGeminiVersion"] = "gemini-2.0-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash-lite"

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
#        "jira:SCRUM",
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

    # drive test with list of known data sources
    for fileName in dictDocuments:

        # filter by test list of names
        if fileName in fileList:
            if re.match('jira:', fileName):
                # Jira export processing
                context["JiraExport"] = True
                context["inputFileName"] = fileName[5:]
                context["finalJSON"] = "webapp/indexer/input/" + fileName[5:] + ".json"
                inputFileBaseName = fileName
            else:
                context["inputFileName"] = "webapp/indexer/input/" + fileName
                context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
                context["rawJSON"] = context["inputFileName"] + ".raw.json"
                
                # folder for bm25-sparse files
                context["bm25sJSON"] = context["inputFileName"] + ".bm25s"
                context["finalJSON"] = context["inputFileName"] + ".json"
                inputFileBaseName = str(Path(context["inputFileName"]).name)

            context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
            if dictDocuments[inputFileBaseName]["extract"]:
                context["extractPattern"] = dictDocuments[inputFileBaseName]["extract"]
            if dictDocuments[inputFileBaseName]["assign"]:
                context["assignList"] = dictDocuments[inputFileBaseName]["assign"]
            context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
        
            issueTemplate = ParserClassFactory.factory(context["issueTemplate"])

            indexerWorkflow = IndexerWorkflow(context, logger)

            testRun(context=context, indexerWorkflow=indexerWorkflow, logger=logger, issueTemplate=issueTemplate)

#            thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate,))
#            thread.start()
#            thread.join()


if __name__ == "__main__":
    main()
