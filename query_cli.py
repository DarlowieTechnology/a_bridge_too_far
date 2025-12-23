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
from common import OneResultList
from query_workflow import QueryWorkflow
from parserClasses import ParserClassFactory


def testMatchXSS(oneResultList : OneResultList) -> bool:

    # Titles of nine known XSS issues 
    knownXSSIssues = [
        "Stored XSS in the Title of the ADD NEW PAGE (Medium)",
        "Stored XSS in PDF files",
        "Stored XSS in the ALT Text in the image upload (Medium)",
        "Reflected Cross-Site Scripting",
        "Stored XSS in the Title Text in the image upload (Medium)",
        "Reflected XSS in api.php",
        "Self XSS in table_row_action.php",
        "Stored XSS in uploaded SVG files",
        "Reflected Cross-Site Scripting (XSS)"
    ]

    found = set()
    for item in oneResultList.results_list:
        IssueTemplate = ParserClassFactory.factory(oneResultList.results_list[0].parser_typename)
        oneIssue = IssueTemplate.model_validate_json(item.data)
        for name in knownXSSIssues:
            if oneIssue.title == name:
                found.add(name)
    notfound = []
    for item in knownXSSIssues:
        if item not in found:
            notfound.append(item)
    if len(notfound) :
        print(f"==FAIL: XSS issues NOT found  Error {100*(len(notfound)/len(knownXSSIssues))} % == \n{json.dumps(notfound)}")
    else:
        print(f"==PASS: XSS issues found==\n")
    return len(notfound) == 0


def testRun(context : dict, logger: Logger) :
    """ 
    Test for query stages 
    
    Args:
        context (dict) - all information for test run
        logger (Logger) - application logger
    Returns:
        None
    """

    queryWorkflow = QueryWorkflow(context, logger) 

    if not queryWorkflow.startup():
        return
    msg = f"workflow startup completed."
    queryWorkflow.workerSnapshot(msg)

    save = []

    msg = f"==query==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.preprocessQuery()
    msg = f"==query preprocessed==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    origQuery = queryWorkflow._context['query']
    save.append(queryWorkflow._context['query'])

    if "hyde" in context["queryvectortransforms"]:
        queryWorkflow.hydeQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query HyDE===\n{json.dumps(queryWorkflow._context['query'])}"
        queryWorkflow.workerSnapshot(msg)
        save.append(queryWorkflow._context['query'])

    if "multi" in context["queryvectortransforms"]:
        queryWorkflow._context['query'] = origQuery
        queryWorkflow.multiQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query multi===\n{json.dumps(queryWorkflow._context['query'])}"
        queryWorkflow.workerSnapshot(msg)
        save.append(queryWorkflow._context['query'])

    if "compress" in context["queryvectortransforms"]:
        queryWorkflow._context['query'] = origQuery
        queryWorkflow.compressQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query compressed===\n{json.dumps(queryWorkflow._context['query'])}"
        queryWorkflow.workerSnapshot(msg)
        save.append(queryWorkflow._context['query'])

    if "rewrite" in context["queryvectortransforms"]:
        queryWorkflow._context['query'] = origQuery
        queryWorkflow.rewriteQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query rewrite===\n{json.dumps(queryWorkflow._context['query'])}"
        queryWorkflow.workerSnapshot(msg)
        save.append(queryWorkflow._context['query'])


    queryWorkflow._context['query'] = save
    msg = f"==final query===\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    oneResultList = queryWorkflow.vectorQuery()
    testMatchXSS(oneResultList)

    return


    queryWorkflow.compressQuery()
    msg = f"==query compressed===\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.hydeQuery()
    msg = f"==query HyDE===\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.preprocessQuery()
    msg = f"==query preprocessed==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.compressQuery()
    msg = f"==query compressed===\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)





    return

    save = queryWorkflow._context['query']
    queryWorkflow.tokenizeQuery()
    msg = f"===query tokenized: stop words, no stemmer==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow._context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = False)
    msg = f"===query tokenized: no stop words, no stemmer==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow._context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = False, useStemmer = True)
    msg = f"===query tokenized: no stop words, stemmer==\n{json.dumps(queryWorkflow._context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow._context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = True, useStemmer = True)
    msg = f"===query tokenized: stop words, stemmer==\n{json.dumps(queryWorkflow._context['query'])}"



    queryWorkflow.bm25sQuery()



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
    context['query'] = 'get all\n  \t\t\nXSS\n issues'
#    context["queryvectortransforms"] = ["hyde", "multi", "compress", "rewrite"]
    context["queryvectortransforms"] = ["rewrite"]
    context["querybm25options"] = ["stopwords", "stemmer"]
    context['cutIssueDistance'] = 0.4
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

    testRun(context=context, logger=logger)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()



