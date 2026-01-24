import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path


# local
import darlowie
from discovery_workflow import DiscoveryWorkflow


def testRun(discoveryWorkflow : DiscoveryWorkflow) -> list[str]:
    """ 
    Test for discovery app
    
    Args:
        discoveryWorkflow (DiscoveryWorkflow) - workflow object
    Returns:
        None    
    """
    totalStart = time.time()
    discoveryWorkflow.context["stage"] = "started"

    msg = f"Input file {discoveryWorkflow.context['inputFileName']}"
    discoveryWorkflow.workerSnapshot(msg)

    # ---------------stage read PDF ---------------

    if "loadDocument" in discoveryWorkflow.context and discoveryWorkflow.context["loadDocument"]:
        startTime = time.time()
        textCombined = discoveryWorkflow.loadPDF(discoveryWorkflow.context['inputFileName'])
        with open(discoveryWorkflow.context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)
        endTime = time.time()
        msg = f"Read text from file {discoveryWorkflow.context['inputFileName']}  {(endTime - startTime):9.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)


    # ---------------parse attempt -----------------

    if "parseAttempt" in discoveryWorkflow.context and discoveryWorkflow.context["parseAttempt"]:
        startTime = time.time()
        with open(discoveryWorkflow.context['rawtextfromPDF'], "r", encoding='utf8', errors='ignore') as txtIn:
            textCombined = txtIn.read()
        dictRawIssues, runUsage = discoveryWorkflow.parseAttemptRecordListOllama(textCombined)
        discoveryWorkflow.addUsage(runUsage)
        with open(discoveryWorkflow.context["rawJSON"], "w" , encoding="utf-8", errors="ignore") as rawJSONOut:
            rawJSONOut.writelines(json.dumps(dictRawIssues, indent=2))
        endTime = time.time()
        rawTextFromPDFBaseName = str(Path(discoveryWorkflow.context['rawtextfromPDF']).name)
        msg = f"Discovered records in raw text {rawTextFromPDFBaseName}. Found {len(dictRawIssues)} potential issues. Time: {(endTime - startTime):9.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    if "finalJSONfromRaw" in discoveryWorkflow.context and discoveryWorkflow.context["finalJSONfromRaw"]:
        startTime = time.time()
        if 'dictRawIssues' not in locals():
            # if this is a separate step - read raw JSON into record collection
            with open(discoveryWorkflow.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                dictRawIssues = json.load(jsonIn)
            msg = f"Read {len(dictRawIssues)} raw records from file {discoveryWorkflow.context['inputFileBaseName']}."
            discoveryWorkflow.workerSnapshot(msg)

        for item in dictRawIssues:
            print(item)
            print("===============================")
            oneRecord, runUsage = discoveryWorkflow.parseOneRecordOllama(item)
            discoveryWorkflow.addUsage(runUsage)
            aaa = json.loads(oneRecord)
            print(aaa)
            print("===============================")
            print(json.dumps(aaa, indent=4))
            print("===============================")

            break

        endTime = time.time()

        finalJSONBaseName = str(Path(discoveryWorkflow.context['finalJSON']).name)
        msg = f"Wrote final JSON: <b>{finalJSONBaseName}</b>. {(endTime - startTime):9.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------stage completed ---------------
    totalEnd = time.time()
    discoveryWorkflow.context["stage"] = "completed"
    msg = f"Processing completed. Usage: {discoveryWorkflow.totalUsageFormat()}. Total time {(totalEnd - totalStart):9.2f} seconds."
    discoveryWorkflow.workerSnapshot(msg)





def main():

    context = darlowie.context

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["DISCLIsession_key"])


    context["inputFileName"] = "webapp/indexer/input/AWS_Review.pdf"
    context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
    context["rawJSON"] = context["inputFileName"] + ".raw.json"
    context["finalJSON"] = context["inputFileName"] + ".json"
    context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)

    context["status"] = []
    context["statusFileName"] = context["DISCLIstatus_FileName"]

    context["loadDocument"] = False
    context["parseAttempt"] = False
    context["finalJSONfromRaw"] = True

    # text extraction from PDF
    context["loadDocument"] = False
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    discoverWorkflow = DiscoveryWorkflow(context, logger)
    testRun(discoverWorkflow)


if __name__ == "__main__":
    main()
