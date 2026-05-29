#
# query CLI app
#
import time
import threading
import logging
import argparse
import json
from pprint import pprint


# local
import darlowie
from common import GLOBALPROVIDER, LLMNAMES, CommonCLIArguments, CommonHelper, QUERYTYPES, TOKENIZERTYPES, ConfigCollection, OpenFile, DebugUtils
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

    if len(queryWorkflow.query):
        queryWorkflow._llmModel = queryWorkflow.createOpenAIModel()
        allQueryResults = queryWorkflow.performQueries()
        if allQueryResults:
            # output results files
            print(f"Output file name: {queryWorkflow.outputFileName}")
            with open(queryWorkflow.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
                jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

    queryWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart), ("Usage", queryWorkflow.totalUsageFormat(insertHTML = False) ) ])
    pprint(queryWorkflow.stats)


def main():

    context = darlowie.context

    parser = argparse.ArgumentParser(description="Query CLI")
    parser.add_argument("--provider", help=f"LLM service provider, for full list pass \"--provider ?\"")
    parser.add_argument("--llm", help=f"LLM name, for full list pass \"---llm ?\"")
    parser.add_argument("--verbose", help=f"Verbosity, one of [DEBUG, INFO, WARN, ERROR, CRITICAL]")
    parser.add_argument("--advanced", help=f"Advanced configuration JSON file")
    parser.add_argument("--query", help="User query (for example \"xss issues\" or \"credentials issues\")")
    parser.add_argument("--output", help=f"Output file with search results, default \"{context['INDEXEOutFile']}\"")
    parser.add_argument("--count", help=f"Count of results in output, default {context['QUECLIoutputCount']}")
    parser.add_argument("--showconfiguration", action='store_const', const=True, help="Show workflow configuration")

    args = parser.parse_args()

    # process advanced configuration first, named parameters below supersede advanced configuration
    if args.advanced:
        context = CommonCLIArguments.processAdvanced(args.advanced, context)
    if args.provider:
        context = CommonCLIArguments.processProvider(args.provider, context)
    if args.llm:
        context = CommonCLIArguments.processLLM(args.llm, context)

    if args.verbose:
        context.setdefault('logginglevel', CommonHelper.convertName2LoggingLevel(args.verbose))

    if args.query:
        # combine --query, --advanced values
        querySet = set()
        querySet.add(args.query)

        if "query" in context.keys():
            # process --advanced, could be str or list[str]
            if type(context['query']) == str:
                querySet.add(context['query'])
            else:
                querySet.update(context['query'])

        context["query"] = list(querySet)
        if (not len(context["query"])):
            print("ERROR: Provide --query parameter")
            return

    if args.output:
        context['outputFileName'] = args.output
    else:
        context.setdefault('outputFileName', context['INDEXEOutFile'])

    if args.count:
        context['outputNumber'] = args.output
    else:
        context.setdefault('outputNumber', context['QUECLIoutputCount'])

    if args.showconfiguration:
        showFlag = True
    else:
        showFlag = False

    # components of hybrid search
    context.setdefault("searchSemanticOriginal", True)
    context.setdefault("searchSemanticOriginalCompress", True)
    context.setdefault("searchSemanticHyDE", True)
    context.setdefault("searchSemanticHyDECompress", True)
    context.setdefault("searchSemanticMulti", True)
    context.setdefault("searchSemanticMultiCompress", True)
    context.setdefault("searchSemanticRewrite", True)
    context.setdefault("searchSemanticRewriteCompress", True)
    context.setdefault("searchBM25sOriginal", True)
    context.setdefault("searchBM25sOriginalCompress", True)
    context.setdefault("searchBM25sPrep", True)
    context.setdefault("searchBM25sPrepCompress", True)

    # ------ other configuration parameter
    #

    context.setdefault("semanticMaxCutItemDistance", 1.0)     # distance cut-off for semantic matches
    context.setdefault("semanticRetrieveNumber", 1000)        # maximum number of semantic items to retrieve

#    context.setdefault("queryBM25Options", TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER)
    context.setdefault("queryBM25Options", TOKENIZERTYPES.STOPWORDSEN)

    context.setdefault("bm25sMinCutOffScore", 0.0)            # bm25s score cut-off
    context.setdefault("bm25sRetrieveNumber", 1000)           # maximum number of bm25s items to retrieve
    context.setdefault("queryPreprocess", True)               # call preprocessQuery() after every query transform

    # output some info about command line arguments
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")


    configCollection = ConfigCollection()
    configCollection.configure(context = context)

    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)

    if showFlag:
        queryWorkflow.showConfiguration()

#    testRun(queryWorkflow=queryWorkflow)

    thread = threading.Thread( target=queryWorkflow.threadWorker)
    thread.start()
    thread.join()

if __name__ == "__main__":
    main()

