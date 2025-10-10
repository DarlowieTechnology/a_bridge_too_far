from django.apps import AppConfig
import logging

class IndexerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'indexer'

    def ready(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)

