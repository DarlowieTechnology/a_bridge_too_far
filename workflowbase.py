#
# base class for workflows
#
from logging import Logger
import json


from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction


# local
from common import COLLECTION, ConfigSingleton, OpenFile


class WorkflowBase:
    """
    Base class for indexer and query workflows
    """

    context : dict = Field(..., description="Context dictionary") 
    config : ConfigSingleton = Field(..., description="Configuration class") 
    logger : Logger = Field(..., description="Application logger") 
    chromaClient : ClientAPI = Field(..., description="ChromaDB Persistent Client") 
    embeddingFunction : OllamaEmbeddingFunction = Field(..., description="ChromaDB embedding Function") 
    collections : dict[str, Collection] = Field(..., description="dictionary of ChromaDB collections") 

    def __init__(self, context : dict, logger : Logger, createCollection : bool):
        """
        Constructor for the base workflow object

        Args:
            context (dict) - context data for workflow
            logger (Logger) - created by caller (CLI or web app)
        """
        self.context = context
        self.config = ConfigSingleton()
        self.logger = logger

        self.embeddingFunction  = self.createEmbeddingFunction()
        self.chromaClient = self.openChromaClient()
        self.collections = {}
        if self.chromaClient:
            for coll in list(COLLECTION):
                self.collections[coll.value] = self.openOrCreateCollection(coll.value, createCollection)


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


    def openChromaClient(self) -> ClientAPI :
        """
        Docstring for openChromaClient
        
        :param self: 
        :return: ChromaDB persistent client or None
        :rtype: ClientAPI
        """
        try:
            chromaClient = chromadb.PersistentClient(
                path=self.config.getAbsPath("rag_datapath"),
                settings=Settings(anonymized_telemetry=False),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        except Exception as e:
            msg = f"Error: ChromaDB exception: {e}"
            self.workerError(msg)
            return None

        return chromaClient


    def createEmbeddingFunction(self) -> OllamaEmbeddingFunction :
        """
        Create Ollama-specific embedding function
        
        :param self: Description
        :return: embedding function object
        :rtype: OllamaEmbeddingFunction
        """
        return OllamaEmbeddingFunction(
            model_name=self.config["rag_embed_llm"],
            url=self.config["rag_embed_url"]
        )


    def openOrCreateCollection(self, collectionName : str, createFlag : bool) -> Collection :
        """
        Open existing ChromaDB collection. If fail to open - create new ChromaDB collection
        Process exceptions and use internal logger
        
        :param self: 
        :param collectionName: name of the ChromaDB collection
        :type str
        :param createFlag: if True, missing collection will be created.
        :type createFlag: bool
        :return: ChromaDB collection or None
        :rtype: Collection
        """

        if collectionName in self.collections:
            return self.collections[collectionName]

        try:
            chromaCollection = self.chromaClient.get_collection(
                name=collectionName,
                embedding_function=self.embeddingFunction
            )
            msg = f"Opened collections {collectionName} with {chromaCollection.count()} documents."
            self.workerSnapshot(msg)
        except chromadb.errors.NotFoundError as e:
            if createFlag:
                try:
                    chromaCollection = self.chromaClient.create_collection(
                        name=collectionName,
                        embedding_function=self.embeddingFunction,
                        metadata={ "hnsw:space": self.config["rag_hnsw_space"]  }
                    )
                    msg = f"Created collection {collectionName}"
                    self.workerSnapshot(msg)
                except Exception as e:
                    msg = f"Error: exception creating collection: {e}"
                    self.workerError(msg)
                    raise
            else:
                msg = f"Error: exception opening collection: {e}"
                self.workerError(msg)
                raise
        except Exception as e:
            msg = f"Error: exception opening collection: {e}"
            self.workerError(msg)
            raise

        return chromaCollection


    def workerSnapshot(self, msg : str):
        """
        Logs status and updates status file

        Args:
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            self.logger.info(msg)
            self.context['status'].append(msg)
        with open(self.context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self.context, indent=2)
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
            self.logger.warning(msg)    
            self.context['status'].append(msg)
        self.context['stage'] = 'error'
        with open(self.context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self.context, indent=2)
            jsonOut.write(formattedOut)

