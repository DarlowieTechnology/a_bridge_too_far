import sys
import logging
import json
import re
from pathlib import Path
import threading

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.apps import apps

import genai_prices

# local
sys.path.append("..")
sys.path.append("../..")

from common import QUERYTYPES, TOKENIZERTYPES, COLLECTION, OpenFile
from parserClasses import ParserClassFactory
from query_workflow import QueryWorkflow
from testQueries import TESTSET, TestSetCollection

from .forms import QueryForm, SettingsColumnOne, SettingsColumnTwo, SettingsColumnThree

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

    context = {}

    queryWorkflow = apps.get_app_config("query").queryWorkflow
    queryTransforms = queryWorkflow.context["querytransforms"]

    testGroupName = TestSetCollection().getCurrentTestName()
    if testGroupName == "None":
        testGroupName = ""

    queryForm = QueryForm( initial={"query": testGroupName} )
    context["query"] = queryForm

    context["DatabaseInfo"] = f"Vector database for report issues: {queryWorkflow.collections[COLLECTION.ISSUES.value].count()}"
    context["RAGSettingsDisplay"] = True
    context["cutIssueDistance"] = queryWorkflow.context["cutIssueDistance"]
    context["semanticRetrieveNum"] = queryWorkflow.context["semanticRetrieveNum"]
    context["QUERYTYPESORIGINAL"] = "Execute" if (QUERYTYPES.ORIGINAL in queryTransforms) or (QUERYTYPES.ORIGINALCOMPRESS in queryTransforms) else "Skip"
    context["QUERYTYPESHYDE"] = "Execute" if (QUERYTYPES.HYDE in queryTransforms) or (QUERYTYPES.HYDECOMPRESS in queryTransforms) else "Skip "
    context["QUERYTYPESMULTI"] = "Execute" if (QUERYTYPES.MULTI in queryTransforms) or (QUERYTYPES.MULTICOMPRESS in queryTransforms) else "Skip"
    context["QUERYTYPESREWRITE"] = "Execute" if (QUERYTYPES.REWRITE in queryTransforms) or (QUERYTYPES.REWRITECOMPRESS in queryTransforms) else "Skip"
    context["bm25sCutOffScore"] = queryWorkflow.context["bm25sCutOffScore"]
    context["bm25sRetrieveNum"] = queryWorkflow.context["bm25sRetrieveNum"]
    context["QUERYTYPESBM25SORIG"] = "Execute" if (QUERYTYPES.BM25SORIG in queryTransforms) or (QUERYTYPES.BM25SORIGCOMPRESS in queryTransforms) else "Skip"
    context["QUERYTYPESBM25PREP"] = "Execute" if (QUERYTYPES.BM25PREP in queryTransforms) or (QUERYTYPES.BM25PREPCOMPRESS in queryTransforms) else "Skip"
    context["TOKENIZERTYPESSTOPWORDSEN"] = "Enabled" if TOKENIZERTYPES.STOPWORDSEN in queryWorkflow.context["querybm25options"] else "Disabled"
    context["TOKENIZERTYPESSTEMMER"] = "Enabled" if TOKENIZERTYPES.STEMMER in queryWorkflow.context["querybm25options"] else "Disabled"
    context["queryPreprocess"] = queryWorkflow.context["queryPreprocess"]
    context["queryCompress"] = queryWorkflow.context["queryCompress"]
    context["rrfTopResults"] = queryWorkflow.context["rrfTopResults"]
    context["wellknownTestSet"] = TestSetCollection().getCurrentTestName()

    return render(request, "query/index.html", context)


def settings(request):
    """
    Settings form of query web app
    Accepts QueryWorkflow settings
    
    Args:
        request

    Returns:
        render query/settings.html

    """
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger("query:" + request.session.session_key)
    queryWorkflow = apps.get_app_config("query").queryWorkflow
    queryTransforms = queryWorkflow.context["querytransforms"]

    context = {}
    context["RAGSettingsDisplay"] = True

    if request.method == "GET":

        columnOneForm = SettingsColumnOne(
            initial={"cutIssueDistance": queryWorkflow.context["cutIssueDistance"],
                     "QUERYTYPESORIGINAL": True if (QUERYTYPES.ORIGINAL in queryTransforms) or (QUERYTYPES.ORIGINALCOMPRESS in queryTransforms) else False,
                     "QUERYTYPESHYDE": True if (QUERYTYPES.HYDE in queryTransforms) or (QUERYTYPES.HYDECOMPRESS in queryTransforms) else False,
                     "QUERYTYPESMULTI": True if (QUERYTYPES.MULTI in queryTransforms) or (QUERYTYPES.MULTICOMPRESS in queryTransforms) else False,
                     "QUERYTYPESREWRITE": True if (QUERYTYPES.REWRITE in queryTransforms) or (QUERYTYPES.REWRITECOMPRESS in queryTransforms) else False,
                     "semanticRetrieveNum": queryWorkflow.context["semanticRetrieveNum"]
                     }
        )
        context["columnOne"] = columnOneForm

        columnTwoForm = SettingsColumnTwo(
            initial={"bm25sCutOffScore": queryWorkflow.context["bm25sCutOffScore"],
                     "QUERYTYPESBM25SORIG": True if (QUERYTYPES.BM25SORIG in queryTransforms) or (QUERYTYPES.BM25SORIGCOMPRESS in queryTransforms) else False,
                     "QUERYTYPESBM25PREP": True if (QUERYTYPES.BM25PREP in queryTransforms) or (QUERYTYPES.BM25PREPCOMPRESS in queryTransforms) else False,
                     "bm25sRetrieveNum": queryWorkflow.context["bm25sRetrieveNum"],
                     "TOKENIZERTYPESSTOPWORDSEN": True if TOKENIZERTYPES.STOPWORDSEN in queryWorkflow.context["querybm25options"] else False,
                     "TOKENIZERTYPESSTEMMER": True if TOKENIZERTYPES.STEMMER in queryWorkflow.context["querybm25options"] else False
                    }
        )
        context["columnTwo"] = columnTwoForm

        columnThreeForm = SettingsColumnThree(
               initial={"queryPreprocess": queryWorkflow.context["queryPreprocess"],
                        "queryCompress": queryWorkflow.context["queryCompress"],
                        "rrfTopResults": queryWorkflow.context["rrfTopResults"],
                        "wellknownTestSet": TestSetCollection().getCurrentTestType()
               }
        )
        context["columnThree"] = columnThreeForm

        return render(request, "query/settings.html", context)
    
    elif request.method == "POST":

        if 'save' in request.POST:
            varQUERYTYPESORIGINAL = QUERYTYPES.NONE
            varQUERYTYPESHYDE = QUERYTYPES.NONE
            varQUERYTYPESMULTI = QUERYTYPES.NONE
            varQUERYTYPESREWRITE = QUERYTYPES.NONE
            varQUERYTYPESBM25SORIG = QUERYTYPES.NONE
            varQUERYTYPESBM25PREP = QUERYTYPES.NONE

            varTOKENIZERTYPESSTOPWORDSEN = TOKENIZERTYPES.NONE
            varTOKENIZERTYPESSTEMMER = TOKENIZERTYPES.NONE

            columnOneForm = SettingsColumnOne(request.POST)
            if columnOneForm.is_valid():
                queryWorkflow.context["cutIssueDistance"] = columnOneForm.cleaned_data["cutIssueDistance"]
                varQUERYTYPESORIGINAL = QUERYTYPES.ORIGINAL if columnOneForm.cleaned_data["QUERYTYPESORIGINAL"] else QUERYTYPES.NONE
                varQUERYTYPESHYDE = QUERYTYPES.HYDE if columnOneForm.cleaned_data["QUERYTYPESHYDE"] else QUERYTYPES.NONE
                varQUERYTYPESMULTI = QUERYTYPES.MULTI if columnOneForm.cleaned_data["QUERYTYPESMULTI"] else QUERYTYPES.NONE
                varQUERYTYPESREWRITE = QUERYTYPES.REWRITE if columnOneForm.cleaned_data["QUERYTYPESREWRITE"] else QUERYTYPES.NONE
                queryWorkflow.context["semanticRetrieveNum"] = columnOneForm.cleaned_data["semanticRetrieveNum"]

            columnTwoForm = SettingsColumnTwo(request.POST)
            if columnTwoForm.is_valid():
                queryWorkflow.context["bm25sCutOffScore"] = columnTwoForm.cleaned_data["bm25sCutOffScore"]
                varQUERYTYPESBM25SORIG = QUERYTYPES.BM25SORIG if columnTwoForm.cleaned_data["QUERYTYPESBM25SORIG"] else QUERYTYPES.NONE
                varQUERYTYPESBM25PREP = QUERYTYPES.BM25PREP if columnTwoForm.cleaned_data["QUERYTYPESBM25PREP"] else QUERYTYPES.NONE
                queryWorkflow.context["bm25sRetrieveNum"] = columnTwoForm.cleaned_data["bm25sRetrieveNum"]
                varTOKENIZERTYPESSTOPWORDSEN = TOKENIZERTYPES.STOPWORDSEN if columnTwoForm.cleaned_data["TOKENIZERTYPESSTOPWORDSEN"] else TOKENIZERTYPES.NONE
                varTOKENIZERTYPESSTEMMER = TOKENIZERTYPES.STEMMER if columnTwoForm.cleaned_data["TOKENIZERTYPESSTEMMER"] else TOKENIZERTYPES.NONE

            columnThreeForm = SettingsColumnThree(request.POST)
            if columnThreeForm.is_valid():
                queryWorkflow.context["queryPreprocess"] = columnThreeForm.cleaned_data["queryPreprocess"]
                queryWorkflow.context["queryCompress"] = columnThreeForm.cleaned_data["queryCompress"]
                queryWorkflow.context["rrfTopResults"] = columnThreeForm.cleaned_data["rrfTopResults"]

                wellknownTestSet = columnThreeForm.cleaned_data["wellknownTestSet"]

                if wellknownTestSet == str(TESTSET.NOTEST.value):
                    TestSetCollection().setCurrentTest(TESTSET.NOTEST)
                if wellknownTestSet == str(TESTSET.XSS.value):
                    TestSetCollection().setCurrentTest(TESTSET.XSS)
                if wellknownTestSet == str(TESTSET.CREDS.value):
                    TestSetCollection().setCurrentTest(TESTSET.CREDS)


            if queryWorkflow.context["queryCompress"]:
                varQUERYTYPESORIGINAL = QUERYTYPES.ORIGINALCOMPRESS if varQUERYTYPESORIGINAL != QUERYTYPES.NONE else QUERYTYPES.NONE
                varQUERYTYPESHYDE = QUERYTYPES.HYDECOMPRESS if varQUERYTYPESHYDE != QUERYTYPES.NONE else QUERYTYPES.NONE
                varQUERYTYPESMULTI = QUERYTYPES.MULTICOMPRESS if varQUERYTYPESMULTI != QUERYTYPES.NONE else QUERYTYPES.NONE
                varQUERYTYPESREWRITE = QUERYTYPES.REWRITECOMPRESS if varQUERYTYPESREWRITE != QUERYTYPES.NONE else QUERYTYPES.NONE
                varQUERYTYPESBM25SORIG = QUERYTYPES.BM25SORIGCOMPRESS if varQUERYTYPESBM25SORIG != QUERYTYPES.NONE else QUERYTYPES.NONE
                varQUERYTYPESBM25PREP = QUERYTYPES.BM25PREPCOMPRESS if varQUERYTYPESBM25PREP != QUERYTYPES.NONE else QUERYTYPES.NONE

            queryWorkflow.context["querytransforms"] = varQUERYTYPESORIGINAL | varQUERYTYPESHYDE | varQUERYTYPESMULTI | varQUERYTYPESREWRITE | varQUERYTYPESBM25SORIG | varQUERYTYPESBM25PREP

            queryWorkflow.context["querybm25options"] = varTOKENIZERTYPESSTOPWORDSEN | varTOKENIZERTYPESSTEMMER

        elif 'cancel' in request.POST:
            pass

        elif 'default' in request.POST:
            queryWorkflow.context["cutIssueDistance"] = 0.5
            queryWorkflow.context["semanticRetrieveNum"] = 1000            
            queryWorkflow.context["bm25sCutOffScore"] = 0
            queryWorkflow.context["bm25sRetrieveNum"] = 50
            queryWorkflow.context["queryPreprocess"] = True
            queryWorkflow.context["queryCompress"] = False
            queryWorkflow.context["rrfTopResults"] = 50
            queryWorkflow.context["querytransforms"] = QUERYTYPES.ORIGINAL | QUERYTYPES.HYDE | QUERYTYPES.MULTI | QUERYTYPES.REWRITE | QUERYTYPES.BM25SORIG | QUERYTYPES.BM25PREP
            queryWorkflow.context["querybm25options"] = TOKENIZERTYPES.STOPWORDSEN
            TestSetCollection().setCurrentTest(TESTSET.XSS)

        response = redirect('/query/')
        return response


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

    if request.method == "POST":
        logger.info(f"Query Process POST")
        queryForm = QueryForm(request.POST)

        queryWorkflow = apps.get_app_config("query").queryWorkflow

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

        queryWorkflow.context['query'] = request.POST['query']
        queryWorkflow.context['stage'] = "starting"
        queryWorkflow.context['status'] = []
        queryWorkflow.context['results'] = []

        queryWorkflow.context['statusFileName'] = statusFileName

        msg = f"Starting query app"
        queryWorkflow.workerSnapshot(msg)

        thread = threading.Thread( target=queryWorkflow.threadWorker)
        thread.start()

        context = {}
        context['query'] = request.POST['query']
        context['stage'] = "starting"
        context['status'] = []


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


