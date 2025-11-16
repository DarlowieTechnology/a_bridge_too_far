
from django.shortcuts import render
from django.http import JsonResponse

from typing import List

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb import QueryResult

import pydantic_ai
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

import tomli
import logging
import json
import sys
import re
import time
from datetime import datetime
from pathlib import Path
import threading


from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import genai_prices

# local
sys.path.append("..")
sys.path.append("../..")

from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile
from common import DebugUtils, OneDesc, AllDesc, OneResultList, OneEmployer, AllEmployers
from parserClasses import ParserClassFactory
from indexer_workflow import IndexerWorkflow


def status(request):
    """
    Target of HTTP Async call from indexer/process.html.
    Reads JSON status file and responds to update indexer/process.html.
    
    Args:
        request

    Returns:
        JsonResponse
    """
    statusContext = {}

    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)
    statusFileName = "status.indexer." + request.session.session_key + ".json"
    try:
        with open(statusFileName, "r") as jsonIn:
            statusContext = json.load(jsonIn)
    except Exception as e:
        errorMsg = f"Status Page: status file error {e}"
        logger.info(errorMsg)
        statusContext['status'] = errorMsg
        return JsonResponse(statusContext)
    
    msg = f"Status: Opened {statusFileName}"
    logger.info(msg)
    return JsonResponse(statusContext)



def index(request):
    """
    Front page of indexer web app with the form. 
    Open documents.json and show list of known data sources.
    Accept chosen data source name.
    Pass local context dict to renderer. IndexerWorkflow is not created yet.
    
    Args:
        request

    Returns:
        render indexer/index.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)
    logger.info(f"Starting session")

    localContext = {}

    # read and display known data sources
    with open("indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    localContext["filelist"] = []
    for fileName in dictDocuments:
        localContext["filelist"].append(fileName)
    return render(request, "indexer/index.html", localContext)


def process(request):
    """
    Target of HTTP POST from indexer/index.html.
    Starts workflow.
    
    Args:
        request

    Returns:
        render indexer/process.html

    """

    if not request.session.session_key:
        request.session.create() 

    logger = logging.getLogger("indexer:" + request.session.session_key)
    logger.info(f"Process: Serving POST")

    statusFileName = "status.indexer." + request.session.session_key + ".json"
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                logger.info(f"Process: Removing completed session file {statusFileName}")
            else:    
                logger.info(f"Process: Existing async processing found : {statusFileName}")
                return render(request, "indexer/process.html", context)
        except:
            logger.info(f"Process: Removing corrupt session file : {statusFileName}")

    # read known data sources
    with open("indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    context = {}
    context['stage'] = "starting"
    context['session_key'] = request.session.session_key
    context['statusFileName'] = statusFileName
    context["llmrequests"] = 0
    context['llmrequesttokens'] = 0
    context['llmresponsetokens'] = 0
    context['llmProvider'] = "Gemini"
#    context["llmGeminiVersion"] = "gemini-2.0-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash"
    context["llmGeminiVersion"] = "gemini-2.5-flash-lite"
    context['status'] = []
    context["issuePattern"] = None
    context["issueTemplate"] = None
    context["JiraExport"] = False

    if re.match('jira:', request.POST['filename']):
        # Jira export processing
        context["JiraExport"] = True
        context["inputFileName"] = request.POST['filename'][5:]
        context["finalJSON"] = "indexer/input/" + request.POST['filename'][5:] + ".json"
        inputFileBaseName = request.POST['filename']
    else:
        context["inputFileName"] = "indexer/input/" + request.POST['filename']
        context["rawtextfromPDF"] = context["inputFileName"] + ".raw.txt"
        context["rawJSON"] = context["inputFileName"] + ".raw.json"
        context["finalJSON"] = context["inputFileName"] + ".json"
        inputFileBaseName = str(Path(context["inputFileName"]).name)

    if inputFileBaseName in dictDocuments:
        context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
        context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
    else:
        logger.error(f"ERROR: no definition for document {inputFileBaseName}")
        return render(request, "indexer/process.html", context)

    logger.info(f"Serving POST {inputFileBaseName}")

    issueTemplate = ParserClassFactory.factory(context["issueTemplate"])

    indexerWorkflow = IndexerWorkflow(context, logger)
    msg = f"Starting indexer"
    indexerWorkflow.workerSnapshot(msg)

    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate,))
    thread.start()
    return render(request, "indexer/process.html", context)


# use this to display genAI pricing
#   
def results(request):
    """
    Target of HTTP GET from indexer/process.html.
    Displays API costs.
    
    Args:
        request

    Returns:
        render indexer/results.html

    """

    providers = [
        { "provider" : "anthropic", "model": "claude-3-5-haiku-latest"  },
        { "provider" : "aws", "model": "nova-pro-v1" },
        { "provider" : "azure", "model": "gpt-4" },
        { "provider" : "deepseek", "model": "deepseek-chat" },
        { "provider" : "google", "model": "gemini-pro-1.5" },
        { "provider" : "openai", "model": "gpt-4" },
        { "provider" : "openrouter", "model": "gpt-4" },
        { "provider" : "x-ai", "model": "grok-3" },
        { "provider" : "x-ai", "model": "grok-4-0709" }
    ]

    context = {}
    context["totalrequests"] = request.GET["totalrequests"]
    context["totalrequesttokens"] = request.GET["totalrequesttokens"]
    context["totalresponsetokens"] = request.GET["totalresponsetokens"]

    context["llminfo"] = []
    for providerInfo in providers:
        price_data = genai_prices.calc_price(
            genai_prices.Usage(input_tokens=int(context["totalrequesttokens"]), output_tokens=int(context["totalresponsetokens"])),
            model_ref= providerInfo["model"],
            provider_id = providerInfo["provider"]
        )
        item = {}
        item["provider"] = providerInfo["provider"]
        item["model"] = providerInfo["model"]
        item["costusd"] = f"{price_data.total_price:.4f}"
        audValue = float(price_data.total_price) * 1.53
        item["costaud"] = f"{audValue:.4f}"

        context["llminfo"].append(item)

    return render(request, "indexer/results.html", context)


