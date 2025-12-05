from django.apps import AppConfig
import logging


class QueryConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'query'

    def ready(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
