#
# Indexer CLI app
#

import sys
import logging
import threading
import json
from pathlib import Path


# local
from common import ConfigSingleton, DebugUtils, ReportIssue, AllReportIssues, OpenFile
from indexer_workflow import IndexerWorkflow


def testRun(context : dict) :
    """ 
    Test for indexer stages 
    
    Args:
        context (dict) - all information for test run
    Returns:
        None
    
    """
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])

    indexerWorkflow = IndexerWorkflow(context, logger)
    
    boolResult, sessionInfoOrError = OpenFile.open(context["statusFileName"], True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                logger.info("Process: Removing completed session file")
            else:    
                logger.info("Process: Existing async processing found - exiting")
                return
        except:
            logger.info("Process: Removing corrupt session file")


    #-------- load PDF

    textCombined = indexerWorkflow.loadPDF(context['inputFileName'])
    with open(context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
        rawOut.write(textCombined)

    inputFileBaseName = str(Path(context['inputFileName']).name)
    msg = f"Read input document {inputFileBaseName}"
    indexerWorkflow.workerSnapshot(msg)

    #-------- preprocess

    allReportIssues = AllReportIssues()
    pattern = allReportIssues.pattern
    dictIssues = indexerWorkflow.preprocessReportRawText(textCombined, pattern)
    with open(context['rawJSON'], "w") as jsonOut:
        jsonOut.writelines(json.dumps(dictIssues, indent=2))

    rawJSONBaseName = str(Path(context['rawJSON']).name)
    msg = f"Preprocessed raw text {rawJSONBaseName}. Found {len(dictIssues)} potential issues."
    indexerWorkflow.workerSnapshot(msg)

    #--------- parse ReportIssue records

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    allReportIssues = indexerWorkflow.parseAllIssues(context['inputFileName'], dictIssues, ReportIssue)
    with open(context["finalJSON"], "w") as jsonOut:
        jsonOut.writelines(allReportIssues.model_dump_json(indent=2))

    finalJSONBaseName = str(Path(indexerWorkflow._context['finalJSON']).name)
    msg = f"Fetched {len(allReportIssues.issue_dict)} Wrote final JSON {finalJSONBaseName}."
    indexerWorkflow.workerSnapshot(msg)

    #--------- vectorize

    indexerWorkflow.vectorize(allReportIssues)

    msg = f"Added {len(allReportIssues.issue_dict)} to vector collections ISSUES."
    indexerWorkflow.workerSnapshot(msg)


    context["stage"] = "completed"
    msg = f"Processing completed."
    indexerWorkflow.workerSnapshot(msg)



def main():

    context = {}
    context["session_key"] = "INDEXER"
    context["statusFileName"] = "status.INDEXER.json"
    context["inputFileName"] = "webapp/indexer/input/test.pdf"
    context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
    context["rawJSON"] = context["inputFileName"] + ".raw.json"
    context["finalJSON"] = context["inputFileName"] + ".json"
    context["llmProvider"] = "Gemini"
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []

    #testRun(context=context)
    #return

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])
    indexerWorkflow = IndexerWorkflow(context, logger)
    thread = threading.Thread( target=indexerWorkflow.threadWorker, kwargs={})
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
