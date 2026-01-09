from django.apps import AppConfig
import logging
import sys

# local
sys.path.append("..")
sys.path.append("../..")

from indexer_workflow import IndexerWorkflow

class IndexerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'indexer'

    indexerWorkflow = None

    def ready(self):
#        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

        context["inputFilePath"] = "indexer/input/"
        context["loadDocument"] = 'Skip'
        context["rawTextFromDocument"] = 'Skip'
        context["rawJSONfromText"] = 'Skip'
        context["finalJSONfromRaw"] = 'Skip'


        self.indexerWorkflow = IndexerWorkflow(context, logger)
