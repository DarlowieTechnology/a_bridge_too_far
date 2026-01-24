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

