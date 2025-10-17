import sys
import re
import time
import logging
from pathlib import Path
import threading

from typing import Optional
from enum import Enum
import json
import tomli

from pydantic import BaseModel, ConfigDict, Field
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader

# local
sys.path.append("..")
sys.path.append("../..")


# local
from common import ConfigSingleton, DebugUtils, ReportIssue, AllReportIssues, OpenFile


def workerSnapshot(logger, fileName, context, msg):
    if msg:
        logger.info(msg)
        context['status'].append(msg)
    with open(fileName, "w") as jsonOut:
        formattedOut = json.dumps(context, indent=2)
        jsonOut.write(formattedOut)
    print(msg)

def workerError(logger, fileName, context, msg):
    logger.info(msg)
    context['status'].append(msg)
    context['stage'] = 'error'
    with open(fileName, "w") as jsonOut:
        formattedOut = json.dumps(context, indent=2)
        jsonOut.write(formattedOut)
    print(msg)


def preprocessReportRawText(rawText : str) -> dict :
    """
    split raw text into pages using issue identifier as a separator. Expected format SS-DDD-DDD
    return dict with issue identifier as key
    """
    regexpString = re.compile(r"\w{2}-\d{3}-\d{3}")
    start = -1
    end = -1
    dictIssues = {}
    prevMatch = None
    for match in re.finditer(regexpString, rawText) :
        end = match.start()
        if start > 0 and end > 0:
            dictIssues[prevMatch.group(0)] = rawText[start:end]
            
        start = match.start()
        prevMatch = match

    # process last issue
    end = len(rawText)
    dictIssues[prevMatch.group(0)] = rawText[start:end]
    return dictIssues


def parseIssue(docs : str) -> tuple[ ReportIssue, Usage] :

    """parse one issue out of text"""

    systemPrompt = f"""
    The prompt contains an issue. Here is the JSON schema for the ReportIssue model you must use as context for what information is expected:
    {json.dumps(ReportIssue.model_json_schema(), indent=2)}
    """
    prompt = f"{docs}"

    ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))

    agent = Agent(ollModel,
                output_type=ReportIssue,
                system_prompt = systemPrompt,
                retries=5,
                output_retries=5)
    try:
        result = agent.run_sync(prompt)
        oneIssue = ReportIssue.model_validate_json(result.output.model_dump_json())
        for attr in oneIssue.__dict__:
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
        runUsage = result.usage()
#        DebugUtils.logPydanticObject(oneIssue, "Issue")
#        print(runUsage)
        return oneIssue, runUsage
    except pydantic_ai.exceptions.UnexpectedModelBehavior:
        print(f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior")
    return None, None


def loadPDF(inputFile) -> str :
    """load PDF and combine all pages"""

    loader = PyPDFLoader(file_path = inputFile, mode = "page" )
    docs = loader.load()
    print(f"document loaded. Pages: {len(docs)}")
    textCombined = ""
    for page in docs:
        textCombined += "\n" + page.page_content
    #print(f"Source Text:\n---\n {textCombined}\n------\n")
    return textCombined


def vectorize(logger, statusFileName, context, allIssues) :
    """add all issues to vector database"""
    try:
        chromaClient = chromadb.PersistentClient(
            path=ConfigSingleton().getAbsPath("rag_datapath"),
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
    except Exception as e:
        msg = f"Error: OpenAI API exception: {e}"
        workerError(logger, statusFileName, context, msg)
        return
    
    ef = OllamaEmbeddingFunction(
        model_name=ConfigSingleton().conf["rag_embed_llm"],
        url=ConfigSingleton().conf["rag_embed_url"]    
    )

    collectionName = "reportissues"
    try:
        chromaReportIssues = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
        msg = f"Opened collections REPORTISSUES with {chromaReportIssues.count()} documents."
        workerSnapshot(logger, statusFileName, context, msg)
    except chromadb.errors.NotFoundError as e:
        try:
            chromaReportIssues = chromaClient.create_collection(
                name=collectionName,
                embedding_function=ef,
                metadata={ "hnsw:space": ConfigSingleton().conf["rag_hnsw_space"]  }
            )
            msg = f"Created collection REPORTISSUES"
            workerSnapshot(logger, statusFileName, context, msg)
        except Exception as e:
            msg = f"Error: exception creating collection REPORTISSUES: {e}"
            workerError(logger, statusFileName, context, msg)
            return

    except Exception as e:
        msg = f"Error: exception opening collection REPORTISSUES: {e}"
        workerError(logger, statusFileName, context, msg)
        return


    ids : list[str] = []
    docs : list[str] = []
    docMetadata : list[str] = []
    embeddings = []

    for key in allIssues.issue_dict:
        reportIssue = allIssues.issue_dict[key]
#        print(f"New record: {reportIssue.model_dump_json(indent=2)}")
        recordHash = hash(reportIssue)
#        print(f"New hash: {recordHash}")
        uniqueId = key
        queryResult = chromaReportIssues.get(ids=[uniqueId])
        if (len(queryResult["ids"])) :

            msg = f"Record found in database {reportIssue.identifier}"
            workerSnapshot(logger, statusFileName, context, msg)

            existingRecordJSON = json.loads(queryResult["documents"][0])
            existingRecord = ReportIssue.model_validate(existingRecordJSON)
#            print(f"Existing record: {existingRecord.model_dump_json(indent=2)}")
            existingHash = hash(existingRecord)
#            print(f"Existing hash: {existingHash}")

            if recordHash == existingHash:
                msg = f"Record hash match for {reportIssue.identifier} - skipping"
                workerSnapshot(logger, statusFileName, context, msg)
                continue
            else:
                msg = f"Record hash different for {reportIssue.identifier}"
                workerSnapshot(logger, statusFileName, context, msg)
                chromaReportIssues.delete(ids=[uniqueId])
                msg = f"Deleted record {reportIssue.identifier}"
                workerSnapshot(logger, statusFileName, context, msg)

        ids.append(uniqueId)
        docs.append(reportIssue.model_dump_json())
        docMetadata.append({ "docName" : recordHash } )
        embeddings.append(ef([reportIssue.description])[0])
        msg = f"Record added in database {reportIssue.identifier}."
        workerSnapshot(logger, statusFileName, context, msg)

    if len(ids):
        chromaReportIssues.add(
            embeddings=embeddings,
            documents=docs,
            ids=ids,
            metadatas=docMetadata
        )



def threadWorker(sessionKey, statusFileName, inputFileName, outputJSONFileName):

    # ---------------stage readpdf ---------------
    logger = logging.getLogger(sessionKey)

    context = {}
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context["stage"] = "readpdf"
    context['status'] = []

    time.sleep(1)

    start = time.time()
    totalStart = start

#    configName = str(Path(__file__).parent.resolve()) + '/../../default.toml'
    configName = '../../default.toml'
    try:
        with open(configName, mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {configName}, exception {e}")
        exit
    textCombined = loadPDF(inputFileName)
    dictIssues = preprocessReportRawText(textCombined)
    end = time.time()
    msg = f"Read and preprocessed input document {inputFileName}. Found {len(dictIssues)} potential issues. Time: {(end-start):9.4f} seconds"
    workerSnapshot(logger, statusFileName, context, msg)

    # ---------------stage fetchissues ---------------

    context["stage"] = "fetchissues"
    start = time.time()

    allIssues = AllReportIssues(name = inputFileName, issue_dict = {})

    for key in dictIssues:
        startOneIssue = time.time()
        oneIssue, usageStats = parseIssue(dictIssues[key])
        allIssues.issue_dict[key] = oneIssue
        endOneIssue = time.time()
        if usageStats:
            context["llmrequests"] += usageStats.requests
            context["llmrequesttokens"] += usageStats.request_tokens
            context["llmresponsetokens"] += usageStats.response_tokens
            msg = f"Fetched issue {key}. {(endOneIssue - startOneIssue):9.4f} seconds. {context["llmrequests"]} LLM requests. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
        else:
            msg = f"Fetched issue {key}. {(endOneIssue - startOneIssue):9.4f} seconds."
        workerSnapshot(logger, statusFileName, context, msg)

    end = time.time()
    msg = f"Fetched {len(allIssues.issue_dict)} issues from {inputFileName}. {(end-start):9.4f} seconds"
    workerSnapshot(logger, statusFileName, context, msg)

    # ---------------JSON --------------

    context["stage"] = "json"
    start = time.time()
    with open(outputJSONFileName, "w") as jsonOut:
        jsonOut.writelines(allIssues.model_dump_json(indent=2))
    end = time.time()
    msg = f"Saved JSON to {outputJSONFileName}. {(end-start):9.4f} seconds"
    workerSnapshot(logger, statusFileName, context, msg)


    # ---------------stage vectorize --------------
    context["stage"] = "vectorize"
    start = time.time()

    vectorize(logger, statusFileName, context, allIssues)

    end = time.time()
    msg = f"Added {len(allIssues.issue_dict)} to vector collections ISSUES. {(end-start):9.4f} seconds"
    workerSnapshot(logger, statusFileName, context, msg)


    # ---------------stage completed ---------------

    context["stage"] = "completed"

    totalEnd = time.time()
    msg = f"Processing completed. Total time {(totalEnd-totalStart):9.4f} seconds. LLM requests: {context["llmrequests"]}. LLM request tokens: {context["llmrequesttokens"]}. LLM response tokens: {context["llmresponsetokens"]}."
    workerSnapshot(logger, statusFileName, context, msg)



def main():

    session_key = "BLAH"
    statusFileName = "status.BLAH.json"
    inputFileName = "input/test.pdf"
    outputJSONFileName = "input/test.json"
    logger = logging.getLogger("BLAH")

    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        contextOld = json.loads(sessionInfoOrError)
        logger.info("Process: Existing async processing found")
        if contextOld["stage"] in ["error", "completed"]:
            logger.info("Process: Removing completed session file")
            print("Process: Removing completed session file")
        else:    
#            return render(request, "generator/process.html", context)
            print("Process: Existing async processing found - exiting")
            return

#    thread = threading.Thread( target=threadWorker, args=(session_key, statusFileName, inputFileName, outputJSONFileName))
#    thread.start()
#    thread.join()

    context = {}
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context["stage"] = "readpdf"
    context['status'] = []

    configName = '../../default.toml'
    try:
        with open(configName, mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {configName}, exception {e}")
        exit

    with open("input/test.json", 'r') as file:
        allIssuesJSON = json.load(file)
    allIssues = AllReportIssues.model_validate(allIssuesJSON)
#    DebugUtils.logPydanticObject(allIssues, "All Issues")
    vectorize(logger, statusFileName, context, allIssues)


if __name__ == "__main__":
    main()
