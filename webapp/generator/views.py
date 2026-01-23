from django.shortcuts import render
from django.http import JsonResponse
from django.apps import apps


import logging
import json
import sys
import threading

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
    generatorWorkflow = apps.get_app_config("generator").generatorWorkflow
    statusFileName = generatorWorkflow.context["statusFileName"]
    try:
        with open(statusFileName, "r") as jsonIn:
            statusContext = json.load(jsonIn)
    except Exception as e:
        errorMsg = f"Status Page: status file error {e}"
        logger.info(errorMsg)
        statusContext['status'] = errorMsg
        return JsonResponse(statusContext)
    
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
    logger = logging.getLogger("generator:" + request.session.session_key)
    generatorWorkflow = apps.get_app_config("generator").generatorWorkflow
    statusFileName = "status.indexer." + request.session.session_key + ".json"

    generatorWorkflow = apps.get_app_config("generator").generatorWorkflow
    generatorWorkflow.context["statusFileName"] = statusFileName


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
    generatorWorkflow = apps.get_app_config("generator").generatorWorkflow
    statusFileName = generatorWorkflow.context["statusFileName"]

    context = {}
    context["session_key"] = request.session.session_key
    context["statusFileName"] = statusFileName

    # pass ad text from the web app form
    context["adtext"] = request.POST['adtext']

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
    context["totalinputtokens"] = request.GET["totalinputtokens"]
    context["totaloutputtokens"] = request.GET["totaloutputtokens"]

    context["llminfo"] = []
    for providerInfo in providers:
        price_data = genai_prices.calc_price(
            genai_prices.Usage(input_tokens=int(context["totalinputtokens"]), output_tokens=int(context["totaloutputtokens"])),
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

