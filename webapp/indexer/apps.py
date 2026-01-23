from django.apps import AppConfig
import logging
import sys

# local
sys.path.append("..")
sys.path.append("../..")

import darlowie
from indexer_workflow import IndexerWorkflow

class IndexerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'indexer'

    indexerWorkflow = None

    def ready(self):
#        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        context = darlowie.context

        context['status'] = []
        context['results'] = []

        context["inputFilePath"] = "indexer/input/"

        # text extraction from PDF
        context["loadDocument"] = False
        context["stripWhiteSpace"] = True
        context["convertToLower"] = True
        context["convertToASCII"] = True
        context["singleSpaces"] = True

        # preprocess text
        context["rawTextFromDocument"] = False

        # create final JSON
        context["finalJSONfromRaw"] = False
        
        # prepare BM25 corpus
        context["prepareBM25corpus"] = False

        # complete BM25 database
        context["completeBM25database"] = False

        # vectorize final JSON
        context["vectorizeFinalJSON"] = False

        # export Jira issues
        context["INDEXEjira_export"] = False

        self.indexerWorkflow = IndexerWorkflow(context, logger)
