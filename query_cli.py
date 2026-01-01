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
from common import OneResultList, OneResultWithType, ResultWithTypeList, OneQuerySemanticAppResult, OneQueryBM25SAppResult, AllQueryAppResults, StatsOnResults
from query_workflow import QueryWorkflow
from parserClasses import ParserClassFactory


# Titles of nine known XSS issues 
knownXSSIssues = [
    "stored xss in the title text in the image upload",         # Refinery-CMS.pdf
    "stored xss in the alt text in the image upload",           # Refinery-CMS.pdf
    "stored xss in refinery cms add new page title",            # Refinery-CMS.pdf
    "reflected cross-site scripting",                           # Web App and Ext Infrastructure Report.pdf
    "reflected xss in api.php",                                 # wikimedia.pdf
    "stored xss in pdf files",                                  # wikimedia.pdf
    "stored xss in uploaded svg files",                         # wikimedia.pdf
    "reflected cross-site scripting (xss)",                     # Web App and Infrastructure and Mobile Report.pdf
    "self xss in table_row_action.php",                         # phpMyAdmin.pdf
    "phpmyadmin content security policy",                       # phpMyAdmin.pdf
    "custom javascript may yield privilege escalation",         # wikimedia.pdf
    "users can inspect each other's personal javascript",       # wikimedia.pdf
    "outdated jenkins software"                                 # CD_and_DevOps Review.pdf
]

def inXSSIssuesSemantic(oneQuerySemanticAppResult : OneQuerySemanticAppResult) -> bool:
    title = oneQuerySemanticAppResult.title.lower()
    for name in knownXSSIssues:
        if title == name:
            return True
    return False


def inXSSIssuesBM25S(oneQuerySemanticAppResult : OneQueryBM25SAppResult) -> bool:
    title = oneQuerySemanticAppResult.title.lower()
    for name in knownXSSIssues:
        if title == name:
            return True
    return False


def testMatchSemanticXSS(allQueryAppResults : AllQueryAppResults) -> bool:

    found = set()
    for key in allQueryAppResults.semantic_dict:
        oneQuerySemanticAppResult = allQueryAppResults.semantic_dict[key]
        title = oneQuerySemanticAppResult.title.lower()
        for name in knownXSSIssues:
            if title == name:
                found.add(name)
#                print(f"{name} {oneQuerySemanticAppResult.distanceSemantic}")
    notfound = []
    for item in knownXSSIssues:
        if item not in found:
            notfound.append(item)
    if len(notfound) :
        print(f"==FAIL: XSS issues NOT found  Error {100*(len(notfound)/len(knownXSSIssues))} % == \n{json.dumps(notfound)}")
    else:
        print(f"==PASS: XSS issues found==\n")
    return len(notfound) == 0


def testMatchBM25sXSS(allQueryAppResults : AllQueryAppResults) -> bool:

    found = set()
    for key in allQueryAppResults.bm25s_dict:
        oneQueryBM25SAppResult = allQueryAppResults.bm25s_dict[key]
        title = oneQueryBM25SAppResult.title.lower()
        for name in knownXSSIssues:
            if title == name:
                found.add(name)
 #               print(f"{name} {oneQueryBM25SAppResult.score}")
    notfound = []
    for item in knownXSSIssues:
        if item not in found:
            notfound.append(item)
    if len(notfound) :
        print(f"==FAIL: XSS issues NOT found  Error {100*(len(notfound)/len(knownXSSIssues))} % == \n{json.dumps(notfound)}")
    else:
        print(f"==PASS: XSS issues found==\n")
    return len(notfound) == 0


def queryBM25S(queryWorkflow : QueryWorkflow, tokenizedQuery : str, allQueryAppResults : AllQueryAppResults, folderName : str):
    allQueryAppResults = queryWorkflow.bm25sQuery(tokenizedQuery, allQueryAppResults, folderName)
    msg = f"bm25s search returned {len(allQueryAppResults.bm25s_dict)} records"
    queryWorkflow.workerSnapshot(msg)

    testMatchBM25sXSS(allQueryAppResults)
    scoresForStats = []
    scoresForXSSset = []
    for key in allQueryAppResults.bm25s_dict:
        oneQueryBM25SAppResult = allQueryAppResults.bm25s_dict[key]
        scoresForStats.append(oneQueryBM25SAppResult.score)
        if inXSSIssuesBM25S(oneQueryBM25SAppResult):
            scoresForXSSset.append(oneQueryBM25SAppResult.score)

    statsOnResults = queryWorkflow.statsOnList(scoresForStats)
    msg = f"BM25S:  Q1: {statsOnResults.q1}, Median (Q2): {statsOnResults.q2}, Q3: {statsOnResults.q3}"
    queryWorkflow.workerSnapshot(msg)
    msg = f"BM25S:  length: {statsOnResults.length}, mean: {statsOnResults.mean}, range: {statsOnResults.range}"
    queryWorkflow.workerSnapshot(msg)

    statsOnXssSet = queryWorkflow.statsOnList(scoresForXSSset)
    msg = f"BM25S:  length: {statsOnXssSet.length}, mean: {statsOnXssSet.mean}, range: {statsOnXssSet.range}"
    queryWorkflow.workerSnapshot(msg)

    for key in allQueryAppResults.bm25s_dict:
        oneQueryBM25SAppResult = allQueryAppResults.bm25s_dict[key]
        if inXSSIssuesBM25S(oneQueryBM25SAppResult):
            qLevel = "Q1"
            if (statsOnResults.q1 < oneQueryBM25SAppResult.score) and (statsOnResults.q2 >= oneQueryBM25SAppResult.score):
                qLevel = "Q2"
            if (statsOnResults.q2 < oneQueryBM25SAppResult.score):
                qLevel = "Q3"
            print(f"{qLevel}  {oneQueryBM25SAppResult.identifier} {oneQueryBM25SAppResult.title.lower()} {oneQueryBM25SAppResult.score}")
        else:
            print(f"NEW: {key} {oneQueryBM25SAppResult.title.lower()} {oneQueryBM25SAppResult.score}")



def querySemantic(queryWorkflow : QueryWorkflow, originalQuery : str, hydeQuery: str, multiQuery : str, compressedQuery : str, rewriteQuery : str, allQueryAppResults : AllQueryAppResults ) :

#    allQueryAppResults = queryWorkflow.vectorQuery(originalQuery, allQueryAppResults)
    allQueryAppResults = queryWorkflow.vectorQuery(hydeQuery, allQueryAppResults)
#    allQueryAppResults = queryWorkflow.vectorQuery(multiQuery, allQueryAppResults)
#    allQueryAppResults = queryWorkflow.vectorQuery(compressedQuery, allQueryAppResults)
#    allQueryAppResults = queryWorkflow.vectorQuery(rewriteQuery, allQueryAppResults)

    msg = f"Semantic search returned {len(allQueryAppResults.semantic_dict)} records"
    queryWorkflow.workerSnapshot(msg)

    testMatchSemanticXSS(allQueryAppResults)
    scoresForStats = []
    scoresForXSSset = []
    for key in allQueryAppResults.semantic_dict:
        oneQuerySemanticAppResult = allQueryAppResults.semantic_dict[key]
        scoresForStats.append(oneQuerySemanticAppResult.distanceSemantic)
        if inXSSIssuesSemantic(oneQuerySemanticAppResult):
            scoresForXSSset.append(oneQuerySemanticAppResult.distanceSemantic)

    statsOnResults = queryWorkflow.statsOnList(scoresForStats)
    msg = f"SEM:  Q1: {statsOnResults.q1}, Median (Q2): {statsOnResults.q2}, Q3: {statsOnResults.q3}"
    queryWorkflow.workerSnapshot(msg)
    msg = f"SEM:  length: {statsOnResults.length}, mean: {statsOnResults.mean}, range: {statsOnResults.range}"
    queryWorkflow.workerSnapshot(msg)

    statsOnXssSet = queryWorkflow.statsOnList(scoresForXSSset)
    msg = f"SEM:  length: {statsOnXssSet.length}, mean: {statsOnXssSet.mean}, range: {statsOnXssSet.range}"
    queryWorkflow.workerSnapshot(msg)


    for key in allQueryAppResults.semantic_dict:
        oneQuerySemanticAppResult = allQueryAppResults.semantic_dict[key]
        if inXSSIssuesSemantic(oneQuerySemanticAppResult):
            qLevel = "Q1"
            if (statsOnResults.q1 < oneQuerySemanticAppResult.distanceSemantic) and (statsOnResults.q2 >= oneQuerySemanticAppResult.distanceSemantic):
                qLevel = "Q2"
            if (statsOnResults.q2 < oneQuerySemanticAppResult.distanceSemantic):
                qLevel = "Q3"
            print(f"{qLevel} {key} {oneQuerySemanticAppResult.title.lower()} {oneQuerySemanticAppResult.distanceSemantic}")



def rrfReRanking(allQueryAppResults : AllQueryAppResults) :
    """
    Reciprocal Rank Fusion (RRF) re-ranking of semantic and bm25s search results
    
    :param allQueryAppResults: results
    :type allQueryAppResults: AllQueryAppResults
    """




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

    originalQuery = queryWorkflow.getQuery()
    msg = f"==query==\n{json.dumps(originalQuery)}"
    queryWorkflow.workerSnapshot(msg)

#    hydeQuery = queryWorkflow.hydeQuery(originalQuery)
    hydeQuery = "Cross-Site Scripting (XSS) issues occur when an attacker injects malicious code, such as JavaScript, into a web application's sensitive data, allowing them to steal or modify user information. To prevent XSS, web developers should validate and sanitize all user input, use Content Security Policy (CSP) headers, and keep dependencies up-to-date to ensure any third-party libraries are not vulnerable to exploits."
    msg = f"==query HyDE===\n{json.dumps(hydeQuery)}"
    queryWorkflow.workerSnapshot(msg)

#    multiQuery = queryWorkflow.multiQuery(originalQuery)
    multiQuery = """['What are the most common types of cross-site scripting (XSS) vulnerabilities?', 'Can you provide information on how to prevent and mitigate Cross Site Scripting attacks?', 'I'm looking for examples of XSS security risks, what should I be aware of?', 'How do I identify potential XSS issues in web applications?', 'What is the impact of not addressing XSS attacks on user data and application security?']"""
#    msg = f"==query Multi===\n{multiQuery}"
#    queryWorkflow.workerSnapshot(msg)

#    rewriteQuery = queryWorkflow.rewriteQuery(originalQuery)
    rewriteQuery = """
Based on the original query "xss issues", I'll provide a rewritten query that may improve retrieval chances:
**Rewritten Query:** "owasp top 10 xss vulnerabilities mitigation"
Here's why I made these changes:
1. **Added specific information**: By mentioning OWASP (Open Web Application Security Project) Top 10, we're targeting a widely-referenced and authoritative resource on web application security.
2. **Specified vulnerability type**: Including "xss" as part of the query ensures that search results focus specifically on Cross-Site Scripting vulnerabilities.
3. **Used relevant keywords**: Adding "vulnerabilities" and "mitigation" covers both the issue itself (identifying XSS vulnerabilities) and the solution (how to mitigate them), increasing the chances of retrieving relevant documents.
This rewritten query should retrieve documents that provide actionable advice, examples, or guidelines for identifying and fixing XSS issues in accordance with OWASP recommendations.
"""
#    rewriteQuery = "owasp top 10 xss vulnerabilities mitigation"
#    msg = f"==query Rewrite===\n{rewriteQuery}"
#    queryWorkflow.workerSnapshot(msg)

    compressedQuery = queryWorkflow.compressQuery(originalQuery)
#    msg = f"==query Compress===\n{compressedQuery}"
#    queryWorkflow.workerSnapshot(msg)
#    compressedQuery = """base original query xss issue provide rewrite query improve retrieval chance  rewrite Query owasp top 10 xss vulnerability mitigation  here make change  1 add specific information mention OWASP Open web Application Security Project Top 10 target widely reference authoritative resource web application security  2 specified vulnerability type include xss part query ensure search result focus specifically Cross site Scripting vulnerability  3 use relevant keyword add vulnerability mitigation cover issue identify XSS vulnerability solution mitigate increase chance retrieve relevant document  rewrite query retrieve document provide actionable advice example guideline identify fix XSS issue accordance OWASP recommendation"""



# -------------------------------------------


#    bm25sQuery = queryWorkflow.prepBM25S(originalQuery)
    bm25sQuery = "['XSS', 'Cross-Site Scripting: A type of web application security vulnerability that allows an attacker to inject malicious code into a vulnerable website, which can then be executed by the user browser.']"
    msg = f"==query prep BM25S===\n{bm25sQuery}"
    queryWorkflow.workerSnapshot(msg)

    tokenizedQuery = queryWorkflow.tokenizeQuery(query = bm25sQuery)
    msg = f"===query tokenized: stop words, no stemmer==\n{json.dumps(tokenizedQuery)}"
    queryWorkflow.workerSnapshot(msg)

    setTokens = set()
    for item in tokenizedQuery[0]:
        setTokens.add(item)
    tokenizedQuery[0] = list(setTokens)
    msg = f"===query tokenized uniq\n{json.dumps(tokenizedQuery)}"
    queryWorkflow.workerSnapshot(msg)

    msg = f"Semantic search cut-off distance {context['cutIssueDistance']}"
    queryWorkflow.workerSnapshot(msg)

    allQueryAppResults = AllQueryAppResults(
        bm25s_dict = {},
        semantic_dict = {},
        queryBM25S = tokenizedQuery,
        querySemantic = originalQuery
    )

    querySemantic(queryWorkflow, originalQuery, hydeQuery, multiQuery, compressedQuery, rewriteQuery, allQueryAppResults)

    queryBM25S(queryWorkflow, tokenizedQuery, allQueryAppResults, context['bm25sIndexFolder'])



    return



#    queryWorkflow.preprocessQuery()
#    msg = f"==query preprocessed==\n{json.dumps(queryWorkflow.context['query'])}"
#    queryWorkflow.workerSnapshot(msg)

    if "hyde" in context["queryvectortransforms"]:
        queryWorkflow.hydeQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query HyDE===\n{json.dumps(queryWorkflow.context['query'])}"
        queryWorkflow.workerSnapshot(msg)

    if "multi" in context["queryvectortransforms"]:
        queryWorkflow.multiQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query multi===\n{json.dumps(queryWorkflow.context['query'])}"
        queryWorkflow.workerSnapshot(msg)

    if "compress" in context["queryvectortransforms"]:
        queryWorkflow.compressQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query compressed===\n{json.dumps(queryWorkflow.context['query'])}"
        queryWorkflow.workerSnapshot(msg)

    if "rewrite" in context["queryvectortransforms"]:
        queryWorkflow.rewriteQuery()
        queryWorkflow.preprocessQuery()
        msg = f"==query rewrite===\n{json.dumps(queryWorkflow.context['query'])}"
        queryWorkflow.workerSnapshot(msg)


    msg = f"==final query===\n{json.dumps(queryWorkflow.context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    resultWithTypeList = queryWorkflow.vectorQuery()
    testMatchXSS(resultWithTypeList)

    return

    save = queryWorkflow.context['query']
    queryWorkflow.tokenizeQuery()
    msg = f"===query tokenized: stop words, no stemmer==\n{json.dumps(queryWorkflow.context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = False)
    msg = f"===query tokenized: no stop words, no stemmer==\n{json.dumps(queryWorkflow.context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = False, useStemmer = True)
    msg = f"===query tokenized: no stop words, stemmer==\n{json.dumps(queryWorkflow.context['query'])}"
    queryWorkflow.workerSnapshot(msg)

    queryWorkflow.context['query'] = save
    queryWorkflow.tokenizeQuery(useStopWords = True, useStemmer = True)
    msg = f"===query tokenized: stop words, stemmer==\n{json.dumps(queryWorkflow.context['query'])}"



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
    context["llmBaseUrl"] = "http://localhost:11434/v1"


    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []
    context['query'] = "xss issues"
#    context["queryvectortransforms"] = ["hyde", "multi", "compress", "rewrite"]
    context["queryvectortransforms"] = ["rewrite"]
    context["querybm25options"] = ["stopwords", "stemmer"]
    context['cutIssueDistance'] = 0.5
    context['bm25sCutOffScore'] = 0.0
    
#    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
#    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(context["session_key"])

    # folder for combined bm25s index - the same for all documents
    context["bm25sIndexFolder"] = "webapp/indexer/input/combined.bm25s"

    # check if workflow is already executed
    if not QueryWorkflow.testLock("status.QUERY.json", logger) : 
        return

    testRun(context=context, logger=logger)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()



