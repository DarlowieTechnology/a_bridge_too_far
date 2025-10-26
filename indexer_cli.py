#
# Indexer CLI app
#

import sys
import logging
import threading
import json


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
    
    # redirect all logs to console
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["session_key"])

    indexerWorkflow = IndexerWorkflow(logger)
    inputFileName = context["inputFileName"]
    statusFileName = context["statusFileName"]
    
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
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

    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context["rawtextfromPDF"] = inputFileName + ".raw.txt"
    context["rawJSON"] = inputFileName + ".raw.json"
    context["finalJSON"] = inputFileName + ".json"

    context["stage"] = "completed"
    context['status'] = []
    indexerWorkflow.workerSnapshot(context, "Starting")

# load PDF
    rawTextFileName = context["rawtextfromPDF"]

    textCombined = indexerWorkflow.loadPDF(context, inputFileName)
    with open(rawTextFileName, "w" , encoding="utf-8", errors="ignore") as rawOut:
        rawOut.write(textCombined)

    rawTextFileName = context["rawtextfromPDF"]
    with open(rawTextFileName, "r" , encoding="utf-8", errors="ignore") as rawIn:
        textCombined = rawIn.read()

#preprocess
    outputRawJSONFileName = context["rawJSON"]
    allReportIssues = AllReportIssues()
    pattern = allReportIssues.pattern
    dictIssues = indexerWorkflow.preprocessReportRawText(textCombined, pattern)
    with open(outputRawJSONFileName, "w") as jsonOut:
        jsonOut.writelines(json.dumps(dictIssues, indent=2))

# parse
    dictIssues = {}
    with open(outputRawJSONFileName, 'r') as file:
        dictIssues = json.load(file)
    allReportIssues = indexerWorkflow.parseAllIssues(inputFileName, dictIssues, context, ReportIssue)

    outputJSONFileName = context["finalJSON"]
    with open(outputJSONFileName, "w") as jsonOut:
        jsonOut.writelines(allReportIssues.model_dump_json(indent=2))

# vectorize
    with open(outputJSONFileName, 'r') as file:
        allIssues = AllReportIssues.model_validate(json.load(file))
#    DebugUtils.logPydanticObject(allIssues, "All Issues")
    indexerWorkflow.vectorize(context, allIssues)

    context["stage"] = "completed"
    msg = f"Processing completed."
    indexerWorkflow.workerSnapshot(context, msg)



def main():

    context = {}
    context["session_key"] = "INDEXER"
    context["statusFileName"] = "status.INDEXER.json"
    context["inputFileName"] = "webapp/indexer/input/test.pdf"
    context["llmProvider"] = "Ollama"

#    testRun(context=context)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["session_key"])
    indexerWorkflow = IndexerWorkflow(logger)
    thread = threading.Thread( target=indexerWorkflow.threadWorker, kwargs={'context': context})
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
