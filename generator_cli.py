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
from common import OneRecord, ConfigSingleton, AllDesc, DebugUtils, ReportIssue, AllReportIssues, OpenFile
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
    logger = logging.getLogger(context["session_key"])

    generatorWorkflow = GeneratorWorkflow(context, logger)
    statusFileName = context["statusFileName"]
    
    # check if the async process exists

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


    #-----------------stage configure

    start = time.time()
    totalStart = start

    config = ConfigSingleton()

    context["stage"] = "configure"
    generatorWorkflow.workerSnapshot(None)

    # read ad text

    boolResult, contentJDOrError = OpenFile.open(filePath = context['adFileName'], readContent = True)
    if not boolResult:
        generatorWorkflow.workerError(context, contentJDOrError)
        return
    jobAdRecord = OneRecord(
        id = "", 
        name=context['adFileName'], 
        description=contentJDOrError
    )
    msg = f"Read in job descriptions from {context['adFileName']}"
    generatorWorkflow.workerSnapshot(msg)

    try:
        chromaClient = chromadb.PersistentClient(
            path=config.getAbsPath("rag_datapath"),
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    except Exception as e:
        msg = f"Error: OpenAI API exception: {e}"
        generatorWorkflow.workerError(msg)
        return

    ef = OllamaEmbeddingFunction(
        model_name=config["rag_embed_llm"],
        url=config["rag_embed_url"]    
    )

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
    msg = f"Opened vector collections ACTIVITY with {chromaActivity.count()} documents, SCENARIO with {chromaScenario.count()} documents. {(end-start):9.4f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage summary

    start = time.time()

    context["stage"] = "summary"
    generatorWorkflow.workerSnapshot(None)

    execSummary, usageStats = generatorWorkflow.extractExecSection(jobAdRecord)
    if not execSummary:
        return

    context['jobtitle'] = execSummary.title
    context['execsummary'] = execSummary.description
    if usageStats:
        context["llmrequests"] = usageStats.requests
        context["llmrequesttokens"] = usageStats.request_tokens
        context["llmresponsetokens"] = usageStats.response_tokens

    end = time.time()

    if usageStats:
        msg = f"Extracted executive summary from job description. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
    else:
        msg = f"Extracted executive summary from job description. {(end-start):9.4f} seconds."
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage extract

    start = time.time()

    context["stage"] = "extract"
    generatorWorkflow.workerSnapshot(None)

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
    if usageStats:
        context["llmrequests"] += usageStats.requests
        context["llmrequesttokens"] += usageStats.request_tokens
        context["llmresponsetokens"] += usageStats.response_tokens

    end = time.time()

    if usageStats:
        msg = f"Extracted {len(oneResultList.results_list)} activities from job description. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
    else:
        msg = f"Extracted {len(oneResultList.results_list)} activities from job description. {(end-start):9.4f} seconds."
    generatorWorkflow.workerSnapshot(msg)

    #--------------stage mapping

    start = time.time()

    context["stage"] = "mapping"
    generatorWorkflow.workerSnapshot(None)

    # ChromaDB calls do not account for LLM usage
    oneResultList = generatorWorkflow.mapToActivity(oneResultList, chromaActivity)

    context['mapped'] = oneResultList.results_list

    end = time.time()

    msg = f"Mapped {len(oneResultList.results_list)} activities to vector database. {(end-start):9.4f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #----------------stage projects

    startAllProjects = time.time()

    context["stage"] = "projects"
    generatorWorkflow.workerSnapshot(None)

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

            if usageStats:
                context["llmrequests"] += usageStats.requests
                context["llmrequesttokens"] += usageStats.request_tokens
                context["llmresponsetokens"] += usageStats.response_tokens

            end = time.time()

            msg = f"Project # {prjCount}: {oneDesc.title}. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
            generatorWorkflow.workerSnapshot(msg)

    endAllProjects = time.time()

    msg = f"Created {len(context['projects'])} projects. {(endAllProjects-startAllProjects):9.4f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    with open(context["adJSONName"], "w") as jsonOut:
        jsonOut.writelines(allDescriptions.model_dump_json(indent=2))

    #--------------stage word

    start = time.time()
    context["stage"] = "word"
    generatorWorkflow.workerSnapshot(None)

    generatorWorkflow.makeWordDoc(allDescriptions)

    end = time.time()
    msg = f"Created Word document {context['wordFileName']}. {(end-start):9.4f} seconds"
    generatorWorkflow.workerSnapshot(msg)

    #--------------stage completed

    totalEnd = time.time()

    context["stage"] = "completed"
    msg = f"Processing completed. Total time {(totalEnd-totalStart):9.4f} seconds. {context["llmrequests"]} LLM requests. {context["llmrequesttokens"]} request tokens. {context["llmresponsetokens"]} response tokens."
    generatorWorkflow.workerSnapshot(msg)




def main():

    context = {}
    context["session_key"] = "GENERATOR"
    context["statusFileName"] = "status." + context["session_key"] + ".json"
    context["adFileName"] = "jobDescriptions/2025-10-02-0001.txt"
    context["adJSONName"] = context["adFileName"] + ".json"
    context["wordFileName"] = context["adFileName"] + ".resume.docx"
    context["llmProvider"] = "Gemini"
    context['status'] = list()
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0


    #testRun(context=context)

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["session_key"])
    generatorWorkflow = GeneratorWorkflow(context, logger)
    thread = threading.Thread( target=generatorWorkflow.threadWorker, kwargs={})
    thread.start()
    thread.join()


if __name__ == "__main__":
    main()
