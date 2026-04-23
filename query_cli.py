#
# query CLI app
#
import time
import threading
import argparse
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
    with open(queryWorkflow.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
        jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

    queryWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart), ("Usage", queryWorkflow.totalUsageFormat(insertHTML = False) ) ])
    pprint(queryWorkflow.stats)




def main():

    context = darlowie.context

    parser = argparse.ArgumentParser(description="Query CLI")
    parser.add_argument("--query", help="User query (\"xss issues\" or \"credentials issues\")")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--count", help="Count of results in output")
    args = parser.parse_args()
    if args.query:
        userQuery = args.query
    else:
        print("Provide user query")
        return
    if args.output:
        outputFileName = args.output
        print(f"Output file name: {outputFileName}")
    else:
        outputFileName = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + "QUERY.results.json"
        print(f"Default output file name: {outputFileName}")
    if args.count:
        outputNumber = args.output
    else:
        outputNumber = 50
        print(f"Default output count: {outputNumber}")


    context['query'] = [userQuery]
#    context['query'] = ["xss issues"]
#    context['query'] = ["credentials issues"]

    context['outputFileName'] = outputFileName
    context['outputNumber'] = outputNumber

    context['status'] = []
    context["statusFileName"] = context["QUECLIstatus_FileName"]
    context['session_key'] = context['QUECLIsession_key']

#    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.ORIGINALCOMPRESS|QUERYTYPES.HYDE|QUERYTYPES.HYDECOMPRESS|QUERYTYPES.MULTI|QUERYTYPES.MULTICOMPRESS|QUERYTYPES.REWRITE|QUERYTYPES.REWRITECOMPRESS|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25SORIGCOMPRESS|QUERYTYPES.BM25PREP|QUERYTYPES.BM25PREPCOMPRESS
    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.HYDE|QUERYTYPES.MULTI|QUERYTYPES.REWRITE|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25PREP
#    context["queryTransforms"] = QUERYTYPES.HYDE


    # other app-specific configuration
    context["dataFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"]
    context["bm25IndexFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + context["QUERYbm25IndexFolder"]

    context['semanticMaxCutItemDistance'] = 1.0       # distance cut-off for semantic matches
    context['semanticRetrieveNumber'] = 1000   # maximum number of semantic items to retrieve

#    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER
    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN

    context['bm25sMinCutOffScore'] = 0.0       # bm25s score cut-off
    context['bm25sRetrieveNumber'] = 1000        # maximum number of bm25s items to retrieve
    
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

