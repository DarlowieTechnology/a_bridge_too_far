from django.apps import AppConfig
import logging
import sys

# local
sys.path.append("..")
sys.path.append("../..")

import darlowie
from generator_workflow import GeneratorWorkflow


class GeneratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'generator'

    generatorWorkflow = None

    def ready(self):
#        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logger = logging.getLogger(__name__)

        context = darlowie.context

        self.generatorWorkflow = GeneratorWorkflow(context, logger)
