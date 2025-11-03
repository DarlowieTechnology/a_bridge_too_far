#
# Indexer CLI app
#

import sys
import logging
import threading
import json
from pathlib import Path

from typing import Any
from typing import List


from pydantic import BaseModel, TypeAdapter, ValidationError

# local
from common import ConfigSingleton, DebugUtils, OpenFile, RecordCollection
from indexer_workflow import IndexerWorkflow
from parserClasses import ReportIssue, ReportFinding, CDReportIssue


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



def testRun(context : dict, issueTemplate: BaseModel) :
    """ 
    Test for indexer stages 
    
    Args:
        context (dict) - all information for test run
        issueTemplate (BaseModel) - template for issue
    Returns:
        None
    
    """
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])
    indexerWorkflow = IndexerWorkflow(context, logger) 
    
    if not testLock(context, logger) : 
        return
    loadPDF(context, indexerWorkflow)
    preprocess(context, indexerWorkflow)
    parseIssues(context, indexerWorkflow, issueTemplate)
    vectorize(context, indexerWorkflow, issueTemplate)

    context["stage"] = "completed"
    msg = f"Processing completed."
    indexerWorkflow.workerSnapshot(msg)


def templateFactory(class_name) -> BaseModel :
    classes = {
        "ReportFinding": ReportFinding,
        "ReportIssue": ReportIssue,
        "CDReportIssue" : CDReportIssue
    }
    return classes[class_name]




def main():

    context = {}
    context["session_key"] = "INDEXER"
    context["statusFileName"] = "status.INDEXER.json"
    context["inputFileName"] = "webapp/indexer/input/Firewall Review.pdf"
    context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
    context["rawJSON"] = context["inputFileName"] + ".raw.json"
    context["finalJSON"] = context["inputFileName"] + ".json"
    context["llmProvider"] = "Gemini"
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []

    # read template description
    with open("webapp/indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    context["issuePattern"] = None
    context["issueTemplate"] = None
    inputFileBaseName = str(Path(context["inputFileName"]).name)
    if inputFileBaseName in dictDocuments:
        context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
        context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]

    issueTemplate = templateFactory(context["issueTemplate"])

    testRun(context=context, issueTemplate=issueTemplate)
    return

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])
    indexerWorkflow = IndexerWorkflow(context, logger)
    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate,))
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
