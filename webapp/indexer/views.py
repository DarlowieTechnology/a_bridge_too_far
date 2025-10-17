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

from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile
from common import DebugUtils, OneDesc, AllDesc, OneResultList, OneEmployer, AllEmployers

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

    return render(request, "indexer/index.html", None)


def process(request):

    context = {}

    logger = logging.getLogger("indexer:" + request.session.session_key)

    # start POST processing
    logger.info(f"Serving POST")
    return render(request, "indexer/process.html", context)

