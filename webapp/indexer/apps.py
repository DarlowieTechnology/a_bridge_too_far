from django.apps import AppConfig
import logging

class IndexerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'indexer'

    def ready(self):
        logging.basicConfig(filename='application.log', level=logging.INFO)
        logger = logging.getLogger(__name__)

