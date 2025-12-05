import sys
import logging
import json
import re
from pathlib import Path
import threading

from django.shortcuts import render
from django.http import JsonResponse

import genai_prices

# local
sys.path.append("..")
sys.path.append("../..")

from common import OpenFile
from parserClasses import ParserClassFactory
from query_workflow import QueryWorkflow



def status(request):
    """
    Target of HTTP Async call from query/process.html.
    Reads JSON status file and responds to update query/process.html.
    
    Args:
        request

    Returns:
        JsonResponse
    """
    statusContext = {}

    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("query:" + request.session.session_key)
    statusFileName = "status.query." + request.session.session_key + ".json"
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
    Front page of query web app with the form. 
    Accept query string
    
    Args:
        request

    Returns:
        render query/index.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("query:" + request.session.session_key)
    logger.info(f"Starting session")

    localContext = {}
    return render(request, "query/index.html", localContext)


def process(request):
    """
    Target of HTTP POST from query/index.html.
    Starts query workflow.
    
    Args:
        request

    Returns:
        render query/process.html

    """

    if not request.session.session_key:
        request.session.create() 

    logger = logging.getLogger("query:" + request.session.session_key)
    logger.info(f"Query Process POST")

    statusFileName = "status.query." + request.session.session_key + ".json"
    logger.info(f"Process: session file name: {statusFileName}")
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                logger.info(f"Process: Removing completed session file {statusFileName}")
            else:    
                logger.info(f"Process: Existing async processing found : {statusFileName}")
                return render(request, "query/process.html", context)
        except:
            logger.info(f"Process: Removing corrupt session file : {statusFileName}")

    # read known data sources
    with open("indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    context = {}
    context['query'] = request.POST['query']
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

    queryWorkflow = QueryWorkflow(context, logger)

    msg = f"Starting query app"
    queryWorkflow.workerSnapshot(msg)

    thread = threading.Thread( target=queryWorkflow.threadWorker)
    thread.start()
    return render(request, "query/process.html", context)

# use this to display genAI pricing
#   
def results(request):
    """
    Target of HTTP GET from query/process.html.
    Displays API costs.
    
    Args:
        request

    Returns:
        render query/results.html

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

    return render(request, "query/results.html", context)

