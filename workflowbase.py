#
# base class for workflows
#

from logging import Logger
import json


from pydantic import BaseModel, Field

# local
from common import ConfigSingleton

class WorkflowBase(BaseModel):

    _context : dict
    _config : ConfigSingleton
    _logger : Logger

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict) - process context data
            logger (Logger) - can originate in CLI or Django app
        """
        self._context = context
        self._config = ConfigSingleton()
        self._logger = logger

    def workerSnapshot(self, msg : str):
        """
        Logs status and updates status file

        Args:
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            self._logger.info(msg)
            self._context['status'].append(msg)
        with open(self._context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self._context, indent=2)
            jsonOut.write(formattedOut)

    def workerError(self, msg : str):
        """
        Logs error and sets process status to error

        Args:
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            self._logger.info(msg)
            self._context['status'].append(msg)
        self._context['stage'] = 'error'
        with open(self._context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self._context, indent=2)
            jsonOut.write(formattedOut)

