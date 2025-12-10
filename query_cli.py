#
# query CLI app
#
import sys
import logging
from logging import Logger
import threading
import json
from pathlib import Path


import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
import pydantic_ai.exceptions
from pydantic import BaseModel, Field
from pydantic_ai.usage import Usage

from openai import OpenAI

# local
from common import OpenFile
from query_workflow import QueryWorkflow
from parserClasses import ParserClassFactory


def testRun(context : dict, queryWorkflow : QueryWorkflow) :
    """ 
    Test for query stages 
    
    Args:
        context (dict) - all information for test run
        queryWorkflow (QueryWorkflow) - query workflow instance
    Returns:
        None
    """

    if not queryWorkflow.startup():
        return

    queryWorkflow.preprocessQuery()

    if context["llmProvider"] == "Gemini":
        queryWorkflow.agentPromptGemini()
    if context["llmProvider"] == "Ollama":
        queryWorkflow.agentPromptOllama()

    context["stage"] = "completed"
    msg = f"Processing completed."
    queryWorkflow.workerSnapshot(msg)


def main():
    context = {}
    context["session_key"] = "QUERY"
    context["statusFileName"] = "status.QUERY.json"
#    context["llmProvider"] = "ChatGPT"
#    context["llmChatGPTVersion"] = "gpt-3.5-turbo"
#    context["llmProvider"] = "Gemini"
#    context["llmGeminiVersion"] = "gemini-2.0-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash-lite"
    context["llmProvider"] = "Ollama"
    context["llmOllamaVersion"] = "llama3.1:latest"


    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []
    context['query'] = 'all\n  \t\t\nXSS\n'
    context['cutIssueDistance'] = 0.50
    context['bm25sCutOffScore'] = 0.0

#    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["session_key"])

    # test list - perform bm25-sparse on data sources from this list
    context["bm25sJSON"] = [
        "webapp/indexer/input/Architecture Review - Threat Model Report.pdf.bm25s",
        "webapp/indexer/input/AWS_Review.pdf.bm25s",
        "webapp/indexer/input/CD_and_DevOps Review.pdf.bm25s",
        "webapp/indexer/input/Database Review.pdf.bm25s",
        "webapp/indexer/input/Firewall Review.pdf.bm25s",
        "webapp/indexer/input/phpMyAdmin.pdf.bm25s",
        "webapp/indexer/input/PHP_Code_Review.pdf.bm25s",
        "webapp/indexer/input/Refinery-CMS.pdf.bm25s",
        "webapp/indexer/input/WASPT_Report.pdf.bm25s",
        "webapp/indexer/input/Web App and Ext Infrastructure Report.pdf.bm25s",
        "webapp/indexer/input/Wikimedia.pdf.bm25s",
        "webapp/indexer/input/Web App and Infrastructure and Mobile Report.pdf.bm25s"
    ]

    # check if workflow is already executed
    if not QueryWorkflow.testLock("status.QUERY.json", logger) : 
        return

    queryWorkflow = QueryWorkflow(context, logger) 

    testRun(context=context, queryWorkflow=queryWorkflow)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()



