from django.apps import AppConfig
import logging

class GeneratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'generator'

    def ready(self):
        logging.basicConfig(filename='application.log', level=logging.INFO)
        logger = logging.getLogger(__name__)
