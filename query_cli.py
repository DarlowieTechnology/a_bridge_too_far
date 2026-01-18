#
# query CLI app
#
import sys
import logging
from logging import Logger
import json
import threading

# local
from common import QUERYTYPES, TOKENIZERTYPES
from resultsQueryClasses import AllQueryResults
from query_workflow import QueryWorkflow
from testQueries import TESTSET, TestSetCollection


def testRun(context : dict, logger: Logger, queryWorkflow : QueryWorkflow) :
    """ 
    Test for query stages 
    
    Args:
        context (dict) - all information for test run
        logger (Logger) - application logger
    Returns:
        None
    """

    if not queryWorkflow.startup():
        msg = f"workflow startup failed."
        queryWorkflow.workerSnapshot(msg)
        return

    allQueryResults = queryWorkflow.performQueries()

    testQuery = TestSetCollection().getCurrentTest()
    for item in allQueryResults.result_lists:
        msg = testQuery.outputRunInfo(item, item.label)
        queryWorkflow.workerSnapshot(msg)

    msg = testQuery.outputRRFInfo(allQueryResults.rrfScores, queryWorkflow.getRRFTopResults())
    queryWorkflow.workerSnapshot(msg)

    score = testQuery.calculateOverallScore(allQueryResults, queryWorkflow.getRRFTopResults()) * 100
    msg = f"Overall score: {score:.4f} %"
    queryWorkflow.workerSnapshot(msg)
    


def main():
    context = {}
    context["session_key"] = "QUERY"
    context["statusFileName"] = "status.QUERY.json"
#    context["llmProvider"] = "Gemini"
#    context["llmVersion"] = "gemini-2.0-flash"
#    context["llmVersion"] = "gemini-2.5-flash"
#    context["llmVersion"] = "gemini-2.5-flash-lite"
    context["llmProvider"] = "Ollama"
    context["llmVersion"] = "llama3.1:latest"
    context["llmBaseUrl"] = "http://localhost:11434/v1"


    context["llmrequests"] = 0
    context["llmrequesttokens"] = 0
    context["llmresponsetokens"] = 0
    context['status'] = []

#    context['query'] = "xss issues"
    context['query'] = "credentials issues"
    TestSetCollection().setCurrentTest(TESTSET.CREDS)

    context["querytransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.HYDE|QUERYTYPES.MULTI|QUERYTYPES.REWRITE|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25PREP
#    context["querytransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.BM25SORIG


    context['cutIssueDistance'] = 0.5       # distance cut-off for semantic matches
    context['semanticRetrieveNum'] = 1000   # maximum number of semantic items to retrieve

#    context["querybm25options"] = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER
    context["querybm25options"] = TOKENIZERTYPES.STOPWORDSEN

    context['bm25sCutOffScore'] = 0.0       # bm25s score cut-off
    context['bm25sRetrieveNum'] = 50        # maximum number of bm25s items to retrieve
    context["bm25sIndexFolder"] = "webapp/indexer/input/combined.bm25s" # # folder for combined bm25s index
    
    context['rrfTopResults'] = 50       # maximum number of RRF results to show

    context['queryPreprocess'] = True         # call preprocessQuery() after every query transform
    context["queryCompress"] = False    # by default Telegraphic Semantic Compression (TSC) is disabled

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(context["session_key"])

    # check if workflow is already executed
    if not QueryWorkflow.testLock("status.QUERY.json", logger) : 
        return

    queryWorkflow = QueryWorkflow(context, logger) 

#    testRun(context=context, logger=logger, queryWorkflow=queryWorkflow)

    thread = threading.Thread( target=queryWorkflow.threadWorker)
    thread.start()
    thread.join()

if __name__ == "__main__":
    main()

