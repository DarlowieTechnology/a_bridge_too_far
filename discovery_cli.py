import sys
import logging
from logging import Logger
import time
import json
from pathlib import Path
import mimetypes

from pydantic_ai.usage import RunUsage

import json_schema_to_pydantic


# local
import darlowie
from common import COLLECTION, RecordCollection
from discovery_workflow import DiscoveryWorkflow

acceptedMimeTypes = [
    "text/css",
    "text/csv",
    "text/html",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/plain"
]


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

    # make data path for the document if does not exist
    Path(discoveryWorkflow.context["dataFolder"]).mkdir(parents=True, exist_ok=True)

    # ---------------loadDocument ---------------

    if "loadDocument" in discoveryWorkflow.context and discoveryWorkflow.context["loadDocument"]:
        startTime = time.time()

        mime_type, encoding = mimetypes.guess_type(discoveryWorkflow.context['inputFileName'])
        if mime_type in acceptedMimeTypes:
            discoveryWorkflow.context['mime_type'] = mime_type
            discoveryWorkflow.context['encoding'] = encoding
        else:
            msg = f" File type not supported: {mime_type}"
            discoveryWorkflow.workerSnapshot(msg)
            return

        if discoveryWorkflow.context['mime_type'] == "application/pdf":
            textCombined = discoveryWorkflow.loadPDF(discoveryWorkflow.context['inputFileName'])
        if discoveryWorkflow.context['mime_type'] in ["text/css", "text/csv", "text/html", "application/json", "text/markdown", "text/plain"]:
            with open(discoveryWorkflow.context["inputFileName"], "r" , encoding="utf-8", errors="ignore") as txtIn:
                textCombined = txtIn.read()

        with open(discoveryWorkflow.context["rawtext"], "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)

        endTime = time.time()
        msg = f"Read <b>{len(textCombined)} bytes</b> from file <b>{discoveryWorkflow.context['inputFileName']}</b>.  Time: {(endTime - startTime):.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------makeSummary -----------------

    if "makeSummary" in discoveryWorkflow.context and discoveryWorkflow.context["makeSummary"]:
        startTime = time.time()
        msg = f"Starting makeSummary phase"
        discoveryWorkflow.workerSnapshot(msg)
        if 'textCombined' not in locals():
            # if this is a separate step - read text into string
            with open(discoveryWorkflow.context['rawtext'], "r", encoding='utf8', errors='ignore') as txtIn:
                textCombined = txtIn.read()

        summaryList, runUsage = discoveryWorkflow.makeSummaryOllama(textCombined)
        discoveryWorkflow.addUsage(runUsage)

        with open(discoveryWorkflow.context["summaryJSON"], "w" , encoding="utf-8", errors="ignore") as summaryJSONOut:
            summaryJSONOut.writelines(json.dumps(summaryList, indent=2))
        endTime = time.time()
        msg = f"Completed makeSummary phase. Usage: {discoveryWorkflow.usageFormat(runUsage)} Time: {(endTime - startTime):.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------attributeCategories ---------

    if "attributeCategories" in discoveryWorkflow.context and discoveryWorkflow.context["attributeCategories"]:
        startTime = time.time()
        msg = f"Starting attributeCategories phase"
        discoveryWorkflow.workerSnapshot(msg)

        if 'summaryList' not in locals():
            # if this is a separate step - read text into string
            with open(discoveryWorkflow.context['summaryJSON'], "r", encoding='utf8', errors='ignore') as JsonIn:
                summaryList = json.load(JsonIn)

#        listRawIssues, runUsage = discoveryWorkflow.parseAttemptRecordListOllama(textCombined)
#        discoveryWorkflow.addUsage(runUsage)

        endTime = time.time()
        msg = f"Completed attributeCategories phase. Time: {(endTime - startTime):.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------parseAttempt -----------------

    if "parseAttempt" in discoveryWorkflow.context and discoveryWorkflow.context["parseAttempt"]:
        startTime = time.time()
        msg = f"Starting parseAttempt phase"
        discoveryWorkflow.workerSnapshot(msg)
        if 'textCombined' not in locals():
            # if this is a separate step - read text into string
            with open(discoveryWorkflow.context['rawtext'], "r", encoding='utf8', errors='ignore') as txtIn:
                textCombined = txtIn.read()
        listRawIssues, runUsage = discoveryWorkflow.parseAttemptRecordListOllama(textCombined)
        discoveryWorkflow.addUsage(runUsage)
        with open(discoveryWorkflow.context["rawJSON"], "w" , encoding="utf-8", errors="ignore") as rawJSONOut:
            rawJSONOut.writelines(json.dumps(listRawIssues, indent=2))
        endTime = time.time()
        msg = f"Completed parseAttempt phase. Found <b>{len(listRawIssues)}</b> potential records. Time: {(endTime - startTime):.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------discoverSchema -----------------

    if "discoverSchema" in discoveryWorkflow.context and discoveryWorkflow.context["discoverSchema"]:
        startTime = time.time()
        msg = f"Starting discoverSchema phase"
        discoveryWorkflow.workerSnapshot(msg)

        if 'listRawIssues' not in locals():
            # if this is a separate step - read raw JSON into list
            with open(discoveryWorkflow.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                listRawIssues = json.load(jsonIn)
            msg = f"Read {len(listRawIssues)} raw records from file {discoveryWorkflow.context['inputFileBaseName']}."
            discoveryWorkflow.workerSnapshot(msg)

        schemaDict, runUsage = discoveryWorkflow.discoverRecordSchemaOllama(listRawIssues)
        if not schemaDict:
            return
        
        schemaDict = json.loads(schemaDict)

        # rename schema to be unique
        schemaDict['title'] = discoveryWorkflow.context["inputFileBaseName"] + " " + schemaDict['title']

        # verify that Pydantic model can be created from schema
        schemaVerified = False
        try:
            PydanticModel = json_schema_to_pydantic.create_model(schemaDict)
            schemaVerified = True
        except json_schema_to_pydantic.SchemaError as e:
            msg = f"Exception SchemaError: {e}"
            discoveryWorkflow.workerSnapshot(msg)
        except json_schema_to_pydantic.TypeError as e:
            msg = f"Exception TypeError: {e}"
            discoveryWorkflow.workerSnapshot(msg)
        except json_schema_to_pydantic.CombinerError as e:
            msg = f"Exception CombinerError: {e}"
            discoveryWorkflow.workerSnapshot(msg)
        except json_schema_to_pydantic.ReferenceError as e:
            msg = f"Exception CombinerError: {e}"
            discoveryWorkflow.workerSnapshot(msg)
        if not schemaVerified:
            return

        # store verified JSON schema
        with open(discoveryWorkflow.context["schemaJSON"], "w", encoding='utf8', errors='ignore') as modelOut:
            modelOut.writelines(json.dumps(schemaDict, indent=2))

        discoveryWorkflow.addUsage(runUsage)
        endTime = time.time()
        msg = f"Completed discoverSchema phase: Usage: {discoveryWorkflow.usageFormat(runUsage)}. Time: {(endTime - startTime):.2f} seconds."
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------finalJSONfromRaw -----------------

    if "finalJSONfromRaw" in discoveryWorkflow.context and discoveryWorkflow.context["finalJSONfromRaw"]:
        startTime = time.time()
        msg = f"Starting finalJSONfromRaw phase"
        discoveryWorkflow.workerSnapshot(msg)
        if 'listRawIssues' not in locals():
            # if this is a separate step - read raw JSON into list
            with open(discoveryWorkflow.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                listRawIssues = json.load(jsonIn)
            msg = f"Read {len(listRawIssues)} raw records from file {discoveryWorkflow.context['inputFileBaseName']}."
            discoveryWorkflow.workerSnapshot(msg)

        if 'PydanticModel' not in locals():
            with open(discoveryWorkflow.context["schemaJSON"], "r", encoding='utf8', errors='ignore') as modelIn:
                schemaDict = json.load(modelIn)
            PydanticModel = json_schema_to_pydantic.create_model(schemaDict)

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
                msg = f"Record: <b>{key}</b>. Usage: {discoveryWorkflow.usageFormat(runUsage)}. Time: {(endOneIssue - startOneIssue):.2f} seconds."
            else:
                msg = f"Record: <b>{key}</b>. {(endOneIssue - startOneIssue):.2f} seconds."
            discoveryWorkflow.workerSnapshot(msg)

        with open(discoveryWorkflow.context["finalJSON"], "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(recordCollection.model_dump_json(indent=2))

        endTime = time.time()
        msg = f"Completed finalJSONfromRaw phase. Usage: {discoveryWorkflow.usageFormat(usageForPhase)}. {(endTime - startTime):.2f} seconds"
        discoveryWorkflow.workerSnapshot(msg)

    # ---------------completed ---------------
    totalEnd = time.time()
    discoveryWorkflow.context["stage"] = "completed"
    msg = f"Workflow completed. Total usage: {discoveryWorkflow.totalUsageFormat()}. Total time {(totalEnd - totalStart):.2f} seconds."
    discoveryWorkflow.workerSnapshot(msg)





def main():

    context = darlowie.context

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["DISCLIsession_key"])


    context["inputFileName"] = "documents/medresearch-099.txt"
    context["inputFileBaseName"] = str(Path(context["inputFileName"]).name)

    context["dataFolder"] = context["inputFileName"] + "-data"
    context["rawtext"] = context["dataFolder"] + "/raw.txt"
    context["rawJSON"] = context["dataFolder"] + "/raw.json"
    context["schemaJSON"] = context["dataFolder"] + "/schema.json"
    context["finalJSON"] = context["dataFolder"] + "/final.json"
    context["summaryJSON"] = context["dataFolder"] + "/summary.json"

    context["status"] = []
    context["statusFileName"] = context["DISCLIstatus_FileName"]

    context["loadDocument"] = True
    context["makeSummary"] = False
    context["attributeCategories"] = False
    context["parseAttempt"] = False
    context["discoverSchema"] = False
    context["finalJSONfromRaw"] = False


    # text extraction configuration from PDF
    context["stripWhiteSpace"] = True
    context["convertToLower"] = True
    context["convertToASCII"] = True
    context["singleSpaces"] = True

    discoverWorkflow = DiscoveryWorkflow(context, logger)
    testRun(discoverWorkflow)


if __name__ == "__main__":
    main()
