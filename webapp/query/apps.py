from django.apps import AppConfig
import logging
import sys

# local
sys.path.append("..")
sys.path.append("../..")

from common import QUERYTYPES, TOKENIZERTYPES
from query_workflow import QueryWorkflow
from testQueries import TESTSET, TestSetCollection


class QueryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'query'

    queryWorkflow = None

    def ready(self):
        #logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        context = {}
        context["llmProvider"] = "Ollama"
        context["llmOllamaVersion"] = "llama3.1:latest"
        context["llmBaseUrl"] = "http://localhost:11434/v1"

        context["llmrequests"] = 0
        context["llmrequesttokens"] = 0
        context["llmresponsetokens"] = 0

        context['status'] = []
        context['results'] = []

        context['query'] = "xss issues"  # default query
        TestSetCollection().setCurrentTest(TESTSET.XSS)

#        context["querytransforms"] = QUERYTYPES.ORIGINAL|QUERYTYPES.HYDE|QUERYTYPES.MULTI|QUERYTYPES.REWRITE|QUERYTYPES.BM25SORIG|QUERYTYPES.BM25PREP
        context["querytransforms"] = QUERYTYPES.ORIGINAL | QUERYTYPES.BM25SORIG

        context['cutIssueDistance'] = 0.5       # distance cut-off for semantic matches
        context['semanticRetrieveNum'] = 1000   # maximum number of semantic items to retrieve

        context["querybm25options"] = TOKENIZERTYPES.STOPWORDSEN    # default options for BM25S
        context['bm25sCutOffScore'] = 0.0       # bm25s score cut-off
        context['bm25sRetrieveNum'] = 50        # maximum number of bm25s items to retrieve
        context["bm25sIndexFolder"] = "indexer/input/combined.bm25s" # # folder for combined bm25s index

        context['rrfTopResults'] = 50       # maximum number of RRF results to show

        context['queryPreprocess'] = True   # by default call preprocessQuery() after every query transform
        context["queryCompress"] = False    # by default Telegraphic Semantic Compression (TSC) is disabled

        self.queryWorkflow = QueryWorkflow(context, logger)






