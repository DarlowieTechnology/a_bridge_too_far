from django.apps import AppConfig
import logging

class GeneratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'generator'

    def ready(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
