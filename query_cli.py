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
from common import OneResultWithType, ResultWithTypeList
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
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])

    if not queryWorkflow.startup():
        return

    if context["llmProvider"] == "ChatGPT":
        queryWorkflow.agentPromptChatGPT()
    if context["llmProvider"] == "Gemini":
        queryWorkflow.agentPromptGemini()

#    while queryWorkflow.prompt("Query or c to cancel:"):
#        print("Continues")

    context["stage"] = "completed"
    msg = f"Processing completed."
    queryWorkflow.workerSnapshot(msg)


def main():
    context = {}
    context["session_key"] = "QUERY"
    context["statusFileName"] = "status.QUERY.json"
#    context["llmProvider"] = "ChatGPT"
#    context["llmChatGPTVersion"] = "gpt-3.5-turbo"
    context["llmProvider"] = "Gemini"
    context["llmGeminiVersion"] = "gemini-2.0-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash"
#    context["llmGeminiVersion"] = "gemini-2.5-flash-lite"

    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])

    queryWorkflow = QueryWorkflow(context, logger) 

    testRun(context=context, queryWorkflow=queryWorkflow)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()




if __name__ == "__main__":
    main()



