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


def workerSnapshot(logger, fileName, context, msg):
    if msg:
        logger.info(msg)
        context['status'].append(msg)
    with open(fileName, "w") as jsonOut:
        formattedOut = json.dumps(context, indent=2)
        jsonOut.write(formattedOut)

def workerError(logger, fileName, context, msg):
    logger.info(msg)
    context['status'].append(msg)
    context['stage'] = 'error'
    with open(fileName, "w") as jsonOut:
        formattedOut = json.dumps(context, indent=2)
        jsonOut.write(formattedOut)


def index(request):
    # create session key and log per session
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

    context = {}
    context['session_key'] = request.session.session_key
    statusFileName = "status." + request.session.session_key + ".json"
    context['statusFileName'] = statusFileName
    context['llmrequesttokens'] = 0
    context['llmresponsetokens'] = 0
    context['llmProvider'] = "Ollama"

    logger = logging.getLogger("indexer:" + request.session.session_key)

    # start POST processing
    fileName = Path("indexer/input/" + request.POST['filename'])
    logger.info(f"Serving POST {fileName}")
    context['inputFileName'] = str(fileName)
    context['rawtextfromPDF'] = str(Path("indexer/input/" + request.POST['filename'] + ".raw.txt"))
    context['rawJSON'] =  str(Path("indexer/input/" + request.POST['filename'] + ".raw.json"))
    context['finalJSON'] = str(Path("indexer/input/" + request.POST['filename'] + ".json"))

    boolResult, sessionInfoOrError = OpenFile.open(context['statusFileName'], True)
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

    msg = f"Processing..."
    logger.info(msg)

    context['status'] = msg
    context['stage'] = 'starting'
    with open(statusFileName, "w") as jsonOut:
        json.dump(context, jsonOut)    

    indexerWorkflow = IndexerWorkflow()
    thread = threading.Thread( target=indexerWorkflow.threadWorker, kwargs={'context': context})
    thread.start()

    return render(request, "indexer/process.html", context)

