#
# query CLI app
#
import time
import threading
import logging
import argparse
from pprint import pprint


# local
import darlowie
from common import GLOBALPROVIDER, LLMNAMES, CommonHelper, QUERYTYPES, TOKENIZERTYPES, ConfigCollection
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
    parser.add_argument("--verbose", help=f"Verbosity, one of [{logging.INFO}, {logging.WARN}]")
    parser.add_argument("--query", help="User query (for example \"xss issues\" or \"credentials issues\")")
    parser.add_argument("--output", help=f"Output file with search results, default \"{defaultOutputFileName}\"")
    parser.add_argument("--count", help=f"Count of results in output, default {context['QUECLIoutputCount']}")

    args = parser.parse_args()

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
        # can be any logging.XXXX values, so we don't check, see Python logging package for details
        context['GLOBALloggerLevel'] = int(args.verbose)

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

    # summary of command line
    print(f"Provider: {context["GLOBALllm_Provider"]}   LLM: {CommonHelper.currentLLMName(context["GLOBALllm_Provider"])}")


    # ------ other configuration parameter
    #
    context['status'] = []
    context["statusFileName"] = context["QUECLIstatus_FileName"]
    context['session_key'] = context['QUECLIsession_key']
    context["ragDatapath"] = context["GLOBALdataFolder"] +  context["QUERYdataFolder"] + context["GLOBALrag_Datapath"]

#    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.ORIGINALCOMPRESS|QUERYTYPES.HYDE|QUERYTYPES.HYDECOMPRESS|QUERYTYPES.MULTI|QUERYTYPES.MULTICOMPRESS|QUERYTYPES.REWRITE|QUERYTYPES.REWRITECOMPRESS|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25SORIGCOMPRESS|QUERYTYPES.BM25PREP|QUERYTYPES.BM25PREPCOMPRESS
    context["queryTransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.HYDE|QUERYTYPES.MULTI|QUERYTYPES.REWRITE|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25PREP
#    context["queryTransforms"] = QUERYTYPES.HYDE


    # other app-specific configuration
    context["dataFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"]
    context["bm25IndexFolder"] = context["GLOBALdataFolder"] + context["QUERYdataFolder"] + context["QUERYbm25IndexFolder"]

    context['semanticMaxCutItemDistance'] = 1.0         # distance cut-off for semantic matches
    context['semanticRetrieveNumber'] = 1000            # maximum number of semantic items to retrieve

#    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER
    context["queryBM25Options"] = TOKENIZERTYPES.STOPWORDSEN

    context['bm25sMinCutOffScore'] = 0.0                # bm25s score cut-off
    context['bm25sRetrieveNumber'] = 1000               # maximum number of bm25s items to retrieve
    
    context['queryPreprocess'] = True         # call preprocessQuery() after every query transform
    context["queryCompress"] = False    # by default Telegraphic Semantic Compression (TSC) is disabled

    configCollection = ConfigCollection(conf = context)
    configCollection.configure()

    queryWorkflow = QueryWorkflow()
    queryWorkflow.configure(configCollection)

    testRun(queryWorkflow=queryWorkflow)

#    thread = threading.Thread( target=queryWorkflow.threadWorker)
#    thread.start()
#    thread.join()

if __name__ == "__main__":
    main()

