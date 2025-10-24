#
# Indexer CLI app
#

import sys
import logging
import threading
import json
import tomli

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
    indexerWorkflow = IndexerWorkflow()
    inputFileName = context["inputFileName"]
    statusFileName = context["statusFileName"]
    logger = logging.getLogger(context["session_key"])

    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                print("Process: Removing completed session file")
            else:    
                print("Process: Existing async processing found - exiting")
                return
        except:
            print("Process: Removing corrupt session file")

    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context["rawtextfromPDF"] = inputFileName + ".raw.txt"
    context["rawJSON"] = inputFileName + ".raw.json"
    context["finalJSON"] = inputFileName + ".json"

    configName = 'default.toml'
    try:
        with open(configName, mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {configName}, exception {e}")
        exit

    context["stage"] = "completed"
    context['status'] = []
    indexerWorkflow.workerSnapshot(logger, statusFileName, context, "Starting")

# load PDF
    rawTextFileName = context["rawtextfromPDF"]

    textCombined = indexerWorkflow.loadPDF(logger, statusFileName, context, inputFileName)
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
    allReportIssues = indexerWorkflow.parseAllIssues(inputFileName, statusFileName, dictIssues, context, logger, ReportIssue)

    outputJSONFileName = context["finalJSON"]
    with open(outputJSONFileName, "w") as jsonOut:
        jsonOut.writelines(allReportIssues.model_dump_json(indent=2))

# vectorize
    with open(outputJSONFileName, 'r') as file:
        allIssues = AllReportIssues.model_validate(json.load(file))
#    DebugUtils.logPydanticObject(allIssues, "All Issues")
    indexerWorkflow.vectorize(logger, statusFileName, context, allIssues)

    context["stage"] = "completed"
    msg = f"Processing completed."
    indexerWorkflow.workerSnapshot(logger, statusFileName, context, msg)



def main():

    context = {}
    context["session_key"] = "BLAH"
    context["statusFileName"] = "status.BLAH.json"
    context["inputFileName"] = "webapp/indexer/input/test.pdf"
    context["llmProvider"] = "Ollama"

    testRun(context=context)

#    indexerWorkflow = IndexerWorkflow()
#    thread = threading.Thread( target=indexerWorkflow.threadWorker, kwargs={'context': context})
#    thread.start()
#    thread.join()


if __name__ == "__main__":
    main()
