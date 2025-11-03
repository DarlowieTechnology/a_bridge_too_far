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

from common import OpenFile
from generator_workflow import GeneratorWorkflow


def status(request):
    """
    Target of HTTP Async call from generator/process.html.
    Reads JSON status file and responds to update generator/process.html.
    
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
    Front page of web app with the form
    
    Args:
        request

    Returns:
        render generator/index.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger(request.session.session_key)
    logger.info(f"Starting session")

    return render(request, "generator/index.html", None)


def process(request):
    """
    Target of HTTP POST from generator/index.html.
    Starts workflow.
    
    Args:
        request

    Returns:
        render generator/process.html

    """

    if not request.session.session_key:
        request.session.create() 

    logger = logging.getLogger("generator:" + request.session.session_key)
    logger.info(f"Process: Serving POST")

    statusFileName = "status.generator." + request.session.session_key + ".json"
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        contextOld = json.loads(sessionInfoOrError)
        logger.info("Process: Existing async processing found")
        if contextOld["stage"] in ["error", "completed"]:
            logger.info("Process: Removing completed session file")
            pass
        else:    
            return render(request, "generator/process.html", context)

    context = {}
    context["session_key"] = request.session.session_key
    context["statusFileName"] = statusFileName

    # pass ad text from the web app form
    context["adtext"] = request.POST['adtext']

    context["adFileName"] = str(request.session.session_key) + ".txt"
    context["adJSONName"] = context["adFileName"] + ".json"
    context["wordFileName"] = context["adFileName"] + ".resume.docx"

    context["llmProvider"] = "Gemini"
    context['status'] = list()
    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0

    generatorWorkflow = GeneratorWorkflow(context, logger)
    msg = f"Starting generator"
    generatorWorkflow.workerSnapshot(msg)

    thread = threading.Thread( target=generatorWorkflow.threadWorker, kwargs={})
    thread.start()

    return render(request, "generator/process.html", context)



# use this to display genAI pricing
#   
def results(request):
    """
    Target of HTTP GET from generator/process.html.
    Displays API costs.
    
    Args:
        request

    Returns:
        render generator/results.html

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

    return render(request, "generator/results.html", context)

