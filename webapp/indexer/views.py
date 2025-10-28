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
    logger = logging.getLogger(request.session.session_key)

    statusFileName = "status." + request.session.session_key + ".json"
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
    Front page of web app with the form. 
    Read file list and pass it to template.
    
    Args:
        request

    Returns:
        render generator/index.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)
    logger.info(f"Starting session")

    context = {}

    data_folder = Path("indexer/input/")
    fileNames = list(data_folder.glob("*.pdf"))
    context["filelist"] = []
    for item in fileNames:
        context["filelist"].append(str(item.name))

    return render(request, "indexer/index.html", context)


def process(request):
    """
    Target of HTTP POST from indexer/index.html.
    Starts workflow.
    
    Args:
        request

    Returns:
        render generator/process.html

    """

    if not request.session.session_key:
        request.session.create() 

    logger = logging.getLogger("indexer:" + request.session.session_key)
    logger.info(f"Process: Serving POST")

    statusFileName = "status." + request.session.session_key + ".json"
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

    fileName = Path("indexer/input/" + request.POST['filename'])
    logger.info(f"Serving POST {fileName}")

    context = {}
    context['session_key'] = request.session.session_key
    context['statusFileName'] = statusFileName
    context['llmrequesttokens'] = 0
    context['llmresponsetokens'] = 0
    context['llmProvider'] = "Ollama"
    context['status'] = list()
    context['inputFileName'] = str(fileName)
    context['rawtextfromPDF'] = str(Path("indexer/input/" + request.POST['filename'] + ".raw.txt"))
    context['rawJSON'] =  str(Path("indexer/input/" + request.POST['filename'] + ".raw.json"))
    context['finalJSON'] = str(Path("indexer/input/" + request.POST['filename'] + ".json"))

    indexerWorkflow = IndexerWorkflow(context, logger)
    thread = threading.Thread( target=indexerWorkflow.threadWorker, kwargs={})
    thread.start()

    return render(request, "indexer/process.html", context)

