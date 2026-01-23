#
# Generator CLI app
#

import sys
import logging
import threading
import json
import time

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction



# local
import darlowie
from common import OneRecord, ConfigSingleton, AllDesc, OpenFile
from generator_workflow import GeneratorWorkflow

def testRun(context : dict) :
    """ 
    Test for generator stages 
    
    Args:
        context (dict) - all information for test run
    Returns:
        None
    
    """
    
    # redirect all logs to console
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["GENCLIsession_key"])

    generatorWorkflow = GeneratorWorkflow(context, logger)
    
    #-----------------stage configure

    start = time.time()
    totalStart = start

    config = ConfigSingleton()

    # read ad text

    boolResult, contentJDOrError = OpenFile.open(filePath = context['GENERAad_FileName'], readContent = True)
    if not boolResult:
        generatorWorkflow.workerError(context, contentJDOrError)
        return
    jobAdRecord = OneRecord(
        id = "", 
        name=context['GENERAad_FileName'], 
        description=contentJDOrError
    )
    msg = f"Read in job descriptions from {context['GENERAad_FileName']}"
    generatorWorkflow.workerSnapshot(msg)

    chromaClient = generatorWorkflow.openChromaClient()
    if not chromaClient:
        return

    ef = generatorWorkflow.createEmbeddingFunction()

    collectionName = "actreal"
    try:
        chromaActivity = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
    except Exception as e:
        msg = f"Error: collection ACTIVITY exception: {e}"
        generatorWorkflow.workerError(msg)
        return

    collectionName = "scenario"
    try:
        chromaScenario = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
    except Exception as e:
        msg = f"Error: collection SCENARIO exception: {e}"
        generatorWorkflow.workerError(msg)
        return

    end = time.time()
    msg = f"Opened collections ACTIVITY with {chromaActivity.count()} documents, SCENARIO with {chromaScenario.count()} documents. Time: {(end-start):9.2f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage summary

    start = time.time()

    execSummary, usageStats = generatorWorkflow.extractExecSection(jobAdRecord)
    if not execSummary:
        return

    context['jobtitle'] = execSummary.title
    context['execsummary'] = execSummary.description
    generatorWorkflow.addUsage(usageStats)

    end = time.time()

    if usageStats:
        msg = f"Extracted executive summary. Usage: {generatorWorkflow.usageFormat(usageStats)}. Time:{(end-start):9.2f} seconds."
    else:
        msg = f"Extracted executive summary. Time: {(end-start):9.2f} seconds."
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage extract

    start = time.time()

    allDescriptions = AllDesc(
        ad_name = jobAdRecord.name,
        exec_section = execSummary,
        project_list = [])

    oneResultList, usageStats = generatorWorkflow.extractInfoFromJobAd(jobAdRecord)

    if not oneResultList:
        msg = f"Internal error on extracting activities from job description"
        generatorWorkflow.workerError(msg)
        return

    context['extracted'] = oneResultList.results_list
    generatorWorkflow.addUsage(usageStats)

    end = time.time()

    if usageStats:
        msg = f"Extracted {len(oneResultList.results_list)} activities. Usage: {generatorWorkflow.usageFormat(usageStats)}. Time: {(end-start):9.2f} seconds."
    else:
        msg = f"Extracted {len(oneResultList.results_list)} activities. Time: {(end-start):9.2f} seconds."
    generatorWorkflow.workerSnapshot(msg)

    #--------------stage mapping

    start = time.time()

    # ChromaDB calls do not account for LLM usage
    oneResultList = generatorWorkflow.mapToActivity(oneResultList, chromaActivity)

    context['mapped'] = oneResultList.results_list

    end = time.time()

    msg = f"Mapped {len(oneResultList.results_list)} activities to database. Time: {(end-start):9.2f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage projects

    startAllProjects = time.time()

    context['projects'] = []
    prjCount = 0
    for chromaQuery in oneResultList.results_list:

        if chromaQuery[:4] == "--- ":
            generatorWorkflow._logger.info(f"!!!!----!!!!!---skipping '{chromaQuery}'")
            continue
        start = time.time()
        oneDesc, usageStats = generatorWorkflow.makeProject(chromaQuery, chromaScenario)
        if oneDesc:
            prjCount += 1
            allDescriptions.project_list.append(oneDesc)
            context['projects'].append(oneDesc.description)
            generatorWorkflow.addUsage(usageStats)

            end = time.time()

            if usageStats:
                msg = f"Project # {prjCount}: {oneDesc.title}. Usage: {generatorWorkflow.usageFormat(usageStats)}. Time: {(end-start):9.2f} seconds."
            else:
                msg = f"Project # {prjCount}: {oneDesc.title}. Time: {(end-start):9.2f} seconds."
            generatorWorkflow.workerSnapshot(msg)

    endAllProjects = time.time()

    msg = f"Created {len(context['projects'])} projects. {(endAllProjects-startAllProjects):9.2f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    with open(context["GENERAad_JSONName"], "w") as jsonOut:
        jsonOut.writelines(allDescriptions.model_dump_json(indent=2))

    #--------------stage word

    start = time.time()

    generatorWorkflow.makeWordDoc(allDescriptions)

    end = time.time()
    msg = f"Created Word document {context['wordFileName']}. {(end-start):9.2f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #--------------stage completed

    totalEnd = time.time()

    msg = f"Processing completed. Usage: {generatorWorkflow.totalUsageFormat()}. Total time {(totalEnd-totalStart):9.2f} seconds."
    generatorWorkflow.workerSnapshot(msg)


def main():

    context = darlowie.context

    context['status'] = []
    context["statusFileName"] = context["GENCLIstatus_FileName"]


    #testRun(context=context)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["GENCLIsession_key"])
    generatorWorkflow = GeneratorWorkflow(context, logger)
    thread = threading.Thread( target=generatorWorkflow.threadWorker, kwargs={})
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
