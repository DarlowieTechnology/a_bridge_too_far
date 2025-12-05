#
# base class for workflows
#

from logging import Logger
import json


from pydantic import BaseModel, Field

# local
from common import ConfigSingleton, OpenFile

class WorkflowBase(BaseModel):

    _context : dict
    _config : ConfigSingleton
    _logger : Logger

    def __init__(self, context : dict, logger : Logger):
        """
        Constructor for the base workflow object
        Args:
            context (dict) - context data for workflow
            logger (Logger) - created by caller (CLI or web app)
        """
        self._context = context
        self._config = ConfigSingleton()
        self._logger = logger

    @staticmethod
    def testLock(statusFileName : str, logger : Logger) -> bool : 
        """
        Status file is used to communicate between workflow thread and CLI, webapp.
        Static method allows to check if status file exists without constructing workflow.
        Args:
            statusFileName (str) - name of status file
            logger (Logger) - created by caller (CLI or web app)
        """
        boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
        if boolResult:
            try:
                contextOld = json.loads(sessionInfoOrError)
                if contextOld["stage"] in ["error", "completed"]:
                    logger.info("Removing completed session file")
                else:    
                    logger.info("Existing instance of workflow found - exiting")
                    return False
            except:
                logger.info("Removing corrupt session file")
        return True


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

