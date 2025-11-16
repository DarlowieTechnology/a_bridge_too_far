import sys
import logging
import json
from pathlib import Path
import threading

from django.shortcuts import render
from django.http import JsonResponse


# local
sys.path.append("..")
sys.path.append("../..")

from common import OpenFile


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

    return render(request, "query/index.html")


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
    logger.info(f"Process: Serving POST")

    statusFileName = "status.query." + request.session.session_key + ".json"
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





