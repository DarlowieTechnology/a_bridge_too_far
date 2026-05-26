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
from common import GLOBALPROVIDER, LLMNAMES, CommonHelper, QUERYTYPES, TOKENIZERTYPES, ConfigCollection, OpenFile, DebugUtils
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

    print(f"Output file name: {queryWorkflow.outputFileName}")
    with open(queryWorkflow.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
        jsonOut.writelines(allQueryResults.model_dump_json(indent=2))

    queryWorkflow.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart), ("Usage", queryWorkflow.totalUsageFormat(insertHTML = False) ) ])
    pprint(queryWorkflow.stats)


def main():

    context = darlowie.context

    defaultOutputFileName = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + "QUERY.results.json"

    parser = argparse.ArgumentParser(description="Query CLI")
    parser.add_argument("--provider", help=f"LLM service provider, for full list pass \"--provider ?\"")
    parser.add_argument("--llm", help=f"LLM name, for full list pass \"---llm ?\"")
    parser.add_argument("--verbose", help=f"Verbosity, one of [DEBUG, INFO, WARN, ERROR, CRITICAL]")
    parser.add_argument("--advanced", help=f"Advanced configuration JSON file")
    parser.add_argument("--query", help="User query (for example \"xss issues\" or \"credentials issues\")")
    parser.add_argument("--output", help=f"Output file with search results, default \"{defaultOutputFileName}\"")
    parser.add_argument("--count", help=f"Count of results in output, default {context['QUECLIoutputCount']}")

    args = parser.parse_args()

    # process advanced configuration first, named parameters below supersede advanced configuration
    if args.advanced:
        res, errOrContent = OpenFile.open(filePath = args.advanced, readContent = True)
        if not res:
            print(errOrContent)
            return
        advDict = json.loads(errOrContent)
        for key in advDict:
            context[key] = advDict[key]

    if args.provider:
        if args.provider == '?':
            CommonHelper.displayProviderLLM(context)
            return
        if args.provider not in GLOBALPROVIDER:
            print(f"Unknown provider {args.provider}")
            return
        else:
            context['GLOBALllm_Provider'] = args.provider

    if args.llm:
        if args.llm == '?':
            CommonHelper.displayProviderLLM(context)
            return
        if args.llm not in LLMNAMES:
            print(f"Unknown LLM {args.llm}")
            return
        else:
            provider = context["GLOBALllm_Provider"]
            CommonHelper.setLLMName(provider, args.llm)

    if args.verbose:
        context['GLOBALloggerLevel'] = DebugUtils.convertName2LoggingLevel(args.verbose)

    if args.query:
        userQuery = args.query
    else:
        print("Provide user query")
        return

    if args.output:
        context['outputFileName'] = args.output
    else:
        context['outputFileName'] = defaultOutputFileName

    if args.count:
        context['outputNumber'] = args.output
    else:
        context['outputNumber'] = context["QUECLIoutputCount"]

    # ------ configurable on command line
    #
    context['query'] = [userQuery]                      # query - configurable on command line
#    context['query'] = ["xss issues"]
#    context['query'] = ["credentials issues"]


    # ------ other configuration parameter
    #
    context['status'] = []

    context['semanticMaxCutItemDistance'] = 1.0         # distance cut-off for semantic matches
    context['semanticRetrieveNumber'] = 1000            # maximum number of semantic items to retrieve

#    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER
    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN

    context['bm25sMinCutOffScore'] = 0.0                # bm25s score cut-off
    context['bm25sRetrieveNumber'] = 1000               # maximum number of bm25s items to retrieve
    
    context['queryPreprocess'] = True         # call preprocessQuery() after every query transform
    context["queryCompress"] = False    # by default Telegraphic Semantic Compression (TSC) is disabled

    # output some info about command line arguments
    print(f"Verbosity level {DebugUtils.convertLoggingLevel2Name(context['GLOBALloggerLevel'])}")
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")


    configCollection = ConfigCollection()
    configCollection.configure(context = context)

    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)

    testRun(queryWorkflow=queryWorkflow)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()

