
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.apps import apps

from typing import List

import logging
import json
import sys
from pathlib import Path
import threading

import genai_prices

# local
sys.path.append("..")
sys.path.append("../..")

from common import OpenFile
from parserClasses import ParserClassFactory

from .forms import IndexerForm, SettingsColumnOne, SettingsColumnTwo, SettingsColumnThree


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
    Pass local context dict to renderer. 
    
    Args:
        request

    Returns:
        render indexer/index.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)

    indexerWorkflow = apps.get_app_config("indexer").indexerWorkflow

    context = {}

    # read and display known data sources
    with open("indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    fileList = []
    for fileName in dictDocuments:
        fileList.append(fileName)

    indexerForm = IndexerForm( context, fileList=fileList )
    context["indexer"] = indexerForm

    context["loadDocument"] = "Execute" if indexerWorkflow.context["loadDocument"] else "Skip"
    context["stripWhiteSpace"] = "Yes" if indexerWorkflow.context["stripWhiteSpace"] else "No"
    context["convertToLower"] = "Yes" if indexerWorkflow.context["convertToLower"] else "No"
    context["convertToASCII"] = "Yes" if indexerWorkflow.context["convertToASCII"] else "No"
    context["singleSpaces"] = "Yes" if indexerWorkflow.context["singleSpaces"] else "No"

    context["rawTextFromDocument"] = "Execute" if indexerWorkflow.context["rawTextFromDocument"] else "Skip"

    context["finalJSONfromRaw"] = "Execute" if indexerWorkflow.context["finalJSONfromRaw"] else "Skip"

    context["prepareBM25corpus"] = "Execute" if indexerWorkflow.context["prepareBM25corpus"] else "Skip"

    context["completeBM25database"] = "Execute" if indexerWorkflow.context["completeBM25database"] else "Skip"

    context["vectorizeFinalJSON"] = "Execute" if indexerWorkflow.context["vectorizeFinalJSON"] else "Skip"

    context["JiraExport"] = "Execute" if indexerWorkflow.context["JiraExport"] else "Skip"

    context["llmProvider"] = indexerWorkflow.context["llmProvider"]
    context["llmVersion"] = indexerWorkflow.context["llmVersion"]

    return render(request, "indexer/index.html", context)


def settings(request):
    """
    Settings form of Indexer web app
    Accepts IndexerWorkflow settings
    
    Args:
        request

    Returns:
        render indexer/settings.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("indexer:" + request.session.session_key)
    indexerWorkflow = apps.get_app_config("indexer").indexerWorkflow

    context = {}

    if request.method == "GET":

        columnOneForm = SettingsColumnOne(
            initial={"LoadDocument": indexerWorkflow.context["loadDocument"],
                     "stripWhiteSpace" : indexerWorkflow.context["stripWhiteSpace"],
                     "convertToLower" : indexerWorkflow.context["convertToLower"],
                     "convertToASCII" : indexerWorkflow.context["convertToASCII"],
                     "singleSpaces" : indexerWorkflow.context["singleSpaces"]
                     }
        )
        context["columnOne"] = columnOneForm

        columnTwoForm = SettingsColumnTwo(
            initial={"rawTextFromDocument": indexerWorkflow.context["rawTextFromDocument"],
                     "finalJSONfromRaw": indexerWorkflow.context["finalJSONfromRaw"],
                     "prepareBM25corpus": indexerWorkflow.context["prepareBM25corpus"],
                     "completeBM25database": indexerWorkflow.context["completeBM25database"]
                    }
        )
        context["columnTwo"] = columnTwoForm

        columnThreeForm = SettingsColumnThree(
               initial={"vectorizeFinalJSON": indexerWorkflow.context["vectorizeFinalJSON"],
                        "JiraExport": indexerWorkflow.context["JiraExport"]
               }
        )
        context["columnThree"] = columnThreeForm

        return render(request, "indexer/settings.html", context)
    
    elif request.method == "POST":

        if 'save' in request.POST:

            columnOneForm = SettingsColumnOne(request.POST)
            if columnOneForm.is_valid():
                indexerWorkflow.context["loadDocument"] = columnOneForm.cleaned_data["LoadDocument"]
                indexerWorkflow.context["stripWhiteSpace"] = columnOneForm.cleaned_data["stripWhiteSpace"]
                indexerWorkflow.context["convertToLower"] = columnOneForm.cleaned_data["convertToLower"]
                indexerWorkflow.context["convertToASCII"] = columnOneForm.cleaned_data["convertToASCII"]
                indexerWorkflow.context["singleSpaces"] = columnOneForm.cleaned_data["singleSpaces"]

            columnTwoForm = SettingsColumnTwo(request.POST)
            if columnTwoForm.is_valid():
                indexerWorkflow.context["rawTextFromDocument"] = columnTwoForm.cleaned_data ["rawTextFromDocument"]
                indexerWorkflow.context["finalJSONfromRaw"] = columnTwoForm.cleaned_data["finalJSONfromRaw"]
                indexerWorkflow.context["prepareBM25corpus"] = columnTwoForm.cleaned_data["prepareBM25corpus"]
                indexerWorkflow.context["completeBM25database"] = columnTwoForm.cleaned_data["completeBM25database"]

            columnThreeForm = SettingsColumnThree(request.POST)
            if columnThreeForm.is_valid():
                indexerWorkflow.context["vectorizeFinalJSON"] = columnThreeForm.cleaned_data["vectorizeFinalJSON"]
                indexerWorkflow.context["JiraExport"] = columnThreeForm.cleaned_data["JiraExport"]

        elif 'cancel' in request.POST:
            pass

        elif 'default' in request.POST:
            indexerWorkflow.context["loadDocument"] = False
            indexerWorkflow.context["stripWhiteSpace"] = True
            indexerWorkflow.context["convertToLower"] = True
            indexerWorkflow.context["convertToASCII"] = True
            indexerWorkflow.context["singleSpaces"] = True
            indexerWorkflow.context["rawTextFromDocument"] = False
            indexerWorkflow.context["finalJSONfromRaw"] = False
            indexerWorkflow.context["prepareBM25corpus"] = False
            indexerWorkflow.context["completeBM25database"] = False
            indexerWorkflow.context["vectorizeFinalJSON"] = False
            indexerWorkflow.context["JiraExport"] = False

        response = redirect('/indexer/')
        return response



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
    indexerWorkflow = apps.get_app_config("indexer").indexerWorkflow


    statusFileName = "status.indexer." + request.session.session_key + ".json"
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        try:
            contextOld = json.loads(sessionInfoOrError)
            if contextOld["stage"] in ["error", "completed"]:
                logger.info(f"Process: Removing completed session file {statusFileName}")
            else:    
                logger.info(f"Process: Existing async processing found : {statusFileName}")
                return render(request, "indexer/process.html", indexerWorkflow.context)
        except:
            logger.info(f"Process: Removing corrupt session file : {statusFileName}")

    # read known data sources
    with open("indexer/input/documents.json", "r", encoding='utf8') as JsonIn:
        dictDocuments = json.load(JsonIn)

    indexerWorkflow.context['stage'] = "starting"
    indexerWorkflow.context['session_key'] = request.session.session_key
    indexerWorkflow.context['statusFileName'] = statusFileName
    indexerWorkflow.context['status'] = []

    if indexerWorkflow.context["JiraExport"]:
        # Jira export processing
        indexerWorkflow.context["inputFileName"] = "SCRUM"
        indexerWorkflow.context["inputFileBaseName"] = "jira:SCRUM"
        indexerWorkflow.context["finalJSON"] = "webapp/indexer/input/SCRUM.json"
        indexerWorkflow.context["issueTemplate"] = "JiraIssueRAG"
    else:
        indexerWorkflow.context["inputFileName"] = "indexer/input/" + request.POST['inputFile']
        indexerWorkflow.context["rawtextfromPDF"] = indexerWorkflow.context["inputFileName"] + ".raw.txt"
        indexerWorkflow.context["rawJSON"] = indexerWorkflow.context["inputFileName"] + ".raw.json"
        indexerWorkflow.context["finalJSON"] = indexerWorkflow.context["inputFileName"] + ".json"
        indexerWorkflow.context["inputFileBaseName"] = str(Path(indexerWorkflow.context["inputFileName"]).name)

        inputFileBaseName = indexerWorkflow.context["inputFileBaseName"]
        if inputFileBaseName in dictDocuments:
            indexerWorkflow.context["issuePattern"] = dictDocuments[inputFileBaseName]["pattern"]
            indexerWorkflow.context["issueTemplate"] = dictDocuments[inputFileBaseName]["templateName"]
            indexerWorkflow.context["extractPattern"] = dictDocuments[inputFileBaseName]["extract"]
            indexerWorkflow.context["assignList"] = dictDocuments[inputFileBaseName]["assign"]
        else:
            logger.error(f"ERROR: no definition for document {inputFileBaseName}")
            return render(request, "indexer/process.html", indexerWorkflow.context)

    issueTemplate = ParserClassFactory.factory(indexerWorkflow.context["issueTemplate"])

    # global corpus for bm25s = updated for every file, completed once for all files
    corpus = []
    thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus))
    thread.start()
    return render(request, "indexer/process.html", indexerWorkflow.context)


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

    return render(request, "indexer/results.html", context)


