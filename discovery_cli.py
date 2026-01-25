import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path

from pydantic_ai.usage import RunUsage

from json_schema_to_pydantic import create_model


# local
import darlowie
from common import COLLECTION, RecordCollection
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
        if 'textCombined' not in locals():
            # if this is a separate step - read text into string
            with open(discoveryWorkflow.context['rawtextfromPDF'], "r", encoding='utf8', errors='ignore') as txtIn:
                textCombined = txtIn.read()
        listRawIssues, runUsage = discoveryWorkflow.parseAttemptRecordListOllama(textCombined)
        discoveryWorkflow.addUsage(runUsage)
        with open(discoveryWorkflow.context["rawJSON"], "w" , encoding="utf-8", errors="ignore") as rawJSONOut:
            rawJSONOut.writelines(json.dumps(listRawIssues, indent=2))
        endTime = time.time()
        rawTextFromPDFBaseName = str(Path(discoveryWorkflow.context['rawtextfromPDF']).name)
        msg = f"Discovered records in raw text {rawTextFromPDFBaseName}. Found {len(listRawIssues)} potential issues. Time: {(endTime - startTime):9.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    if "discoverSchema" in discoveryWorkflow.context and discoveryWorkflow.context["discoverSchema"]:
        startTime = time.time()
        if 'listRawIssues' not in locals():
            # if this is a separate step - read raw JSON into list
            with open(discoveryWorkflow.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                listRawIssues = json.load(jsonIn)
            msg = f"Read {len(listRawIssues)} raw records from file {discoveryWorkflow.context['inputFileBaseName']}."
            discoveryWorkflow.workerSnapshot(msg)
        schemaDict, runUsage = discoveryWorkflow.discoverRecordSchemaOllama(listRawIssues)
        schemaDict = json.loads(schemaDict)
        PydanticModel = create_model(schemaDict)
        with open(discoveryWorkflow.context["schemaJSON"], "w", encoding='utf8', errors='ignore') as modelOut:
            modelOut.writelines(json.dumps(schemaDict, indent=2))
        endTime = time.time()
        if runUsage:
            discoveryWorkflow.addUsage(runUsage)
            msg = f"discoverSchema: {PydanticModel != None}. Usage: {discoveryWorkflow.usageFormat(runUsage)}. Time: {(endTime - startTime):9.2f} seconds."
        else:
            msg = f"discoverSchema: {PydanticModel != None}. {(endTime - startTime):9.2f} seconds."
        discoveryWorkflow.workerSnapshot(msg)

    if "finalJSONfromRaw" in discoveryWorkflow.context and discoveryWorkflow.context["finalJSONfromRaw"]:
        startTime = time.time()
        if 'listRawIssues' not in locals():
            # if this is a separate step - read raw JSON into list
            with open(discoveryWorkflow.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                listRawIssues = json.load(jsonIn)
            msg = f"Read {len(listRawIssues)} raw records from file {discoveryWorkflow.context['inputFileBaseName']}."
            discoveryWorkflow.workerSnapshot(msg)

        if 'PydanticModel' not in locals():
            with open(discoveryWorkflow.context["schemaJSON"], "r", encoding='utf8', errors='ignore') as modelIn:
                schemaDict = json.load(modelIn)
            PydanticModel = create_model(schemaDict)

        recordCollection = RecordCollection(
            report = str(Path(discoveryWorkflow.context['inputFileName']).name),
            finding_dict = {}
        )
        usageForPhase = RunUsage()

        for item in listRawIssues:
            startOneIssue = time.time()
            oneRecord, runUsage = discoveryWorkflow.parseOneRecordOllama(item, PydanticModel)
            key = oneRecord.identifier
            recordCollection[key] = oneRecord
            endOneIssue = time.time()
            if runUsage:
                discoveryWorkflow.addUsage(runUsage)
                usageForPhase += runUsage
                msg = f"Record: <b>{key}</b>. Usage: {discoveryWorkflow.usageFormat(runUsage)}. Time: {(endOneIssue - startOneIssue):9.2f} seconds."
            else:
                msg = f"Record: <b>{key}</b>. {(endOneIssue - startOneIssue):9.2f} seconds."
            discoveryWorkflow.workerSnapshot(msg)

        with open(discoveryWorkflow.context["finalJSON"], "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(recordCollection.model_dump_json(indent=2))

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
    context["schemaJSON"] = context["inputFileName"] + ".schema.json"
    context["finalJSON"] = context["inputFileName"] + ".json"
    context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)

    context["status"] = []
    context["statusFileName"] = context["DISCLIstatus_FileName"]

    context["loadDocument"] = False
    context["parseAttempt"] = False
    context["discoverSchema"] = False
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
