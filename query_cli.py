#
# query CLI app
#
import time
import threading
from pprint import pprint


# local
import darlowie
from common import QUERYTYPES, TOKENIZERTYPES, ConfigCollection
from query_workflow import QueryWorkflow



def testRun(queryWorkflow : QueryWorkflow) :
    """ 
    Test for query stages 
    
    Args:
        context (dict) - all information for test run
        logger (Logger) - application logger
    Returns:
        None
    """

    totalStart = time.time()

    queryWorkflow._llmModel = queryWorkflow.createOpenAIModel()

    allQueryResults = queryWorkflow.performQueries()

    # output results files
    print(f"Query result are written in file: {queryWorkflow.outputFileName}")
    with open(queryWorkflow.outputFileName, "w") as jsonOut:
        jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

    queryWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart), ("Usage", queryWorkflow.totalUsageFormat(insertHTML = False) ) ])
    pprint(queryWorkflow.stats)




def main():
    context = darlowie.context

    context['status'] = []
    context["statusFileName"] = context["QUECLIstatus_FileName"]
    context['session_key'] = context['QUECLIsession_key']


#    context['query'] = "xss issues"
    context['query'] = "credentials issues"

    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.HYDE|QUERYTYPES.MULTI|QUERYTYPES.REWRITE|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25PREP
#    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.BM25SORIG


    # other app-specific configuration
    context["dataFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"]
    context["bm25IndexFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + context["QUERYbm25IndexFolder"]

    context['semanticMaxCutItemDistance'] = 0.5       # distance cut-off for semantic matches
    context['semanticRetrieveNumber'] = 1000   # maximum number of semantic items to retrieve

#    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER
    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN

    context['bm25sMinCutOffScore'] = 0.0       # bm25s score cut-off
    context['bm25sRetrieveNumber'] = 50        # maximum number of bm25s items to retrieve
    
    context['outputNumber'] = 50
    context['outputFileName'] = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + "QUERY.results.json"

    context['queryPreprocess'] = True         # call preprocessQuery() after every query transform
    context["queryCompress"] = False    # by default Telegraphic Semantic Compression (TSC) is disabled

    configCollection = ConfigCollection(context)

    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)

    testRun(queryWorkflow=queryWorkflow)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()

