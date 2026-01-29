#
# base class for workflows
#
from logging import Logger
import json
from typing import List
from pathlib import Path


from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader

from anyascii import anyascii


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
    usage : RunUsage = Field(..., description="LLM usage object")

    def __init__(self, context : dict, logger : Logger, createCollection : bool):
        """
        Constructor for the base workflow object

        Args:
            context (dict) - context data for workflow
            logger (Logger) - created by caller (CLI or web app)
            createCollection (bool) - if True, create vector database
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
        self.usage = RunUsage()


    def openChromaClient(self) -> ClientAPI :
        """
        Opens ChromaDB client for vector database
        
        :return: ChromaDB persistent client or None
        :rtype: ClientAPI
        """
        try:
            chromaClient = chromadb.PersistentClient(
                path=self.getAbsPath("GLOBALrag_datapath"),
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
        
        :return: embedding function object
        :rtype: OllamaEmbeddingFunction
        """
        return OllamaEmbeddingFunction(
            model_name=self.context["GLOBALrag_embed_llm"],
            url=self.context["GLOBALrag_embed_url"]
        )


    def createOpenAIChatModel(self) -> OpenAIChatModel: 
        """
        return OpenAIChatModel class instance
        
        :return: OpenAIChatModel class instance
        :rtype: OpenAIChatModel
        """
        return OpenAIChatModel(model_name=self.context["GLOBALllm_Version"], 
                               provider=OllamaProvider(base_url=self.context["GLOBALllm_base_url"]))


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
#            msg = f"Opened collections {collectionName} with {chromaCollection.count()} documents."
#            self.workerSnapshot(msg)
        except chromadb.errors.NotFoundError as e:
            if createFlag:
                try:
                    chromaCollection = self.chromaClient.create_collection(
                        name=collectionName,
                        embedding_function=self.embeddingFunction,
                        metadata={ "hnsw:space": self.context["GLOBALrag_hnsw_space"]  }
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


    def addUsage(self, newUsage : RunUsage) :
        """
        Add usage to existing total
        
        :param self: Description
        :param newUsage: new RunUsage or None
        :type newUsage: RunUsage
        """
        if newUsage:
            self.usage += newUsage


    def totalUsageFormat(self) -> str:
        """
        Return current usage formatted as string
        
        :return: usage as string
        :rtype: str
        """
        requestLabel = 'requests' if self.usage.requests > 1 else 'request'
        return f"<b>{self.usage.requests}</b> {requestLabel}, <b>{self.usage.input_tokens}>/b> input tokens, <b>{self.usage.output_tokens}</b> output tokens"


    def usageFormat(self, usage : RunUsage) -> str:
        """
        Return usage formatted as string
        
        :return: usage as string
        :rtype: str
        """
        if usage:
            requestLabel = 'requests' if usage.requests > 1 else 'request'
            return f"<b>{usage.requests}</b> {requestLabel}, <b>{usage.input_tokens}</b>, input tokens <b>{usage.output_tokens}</b> output tokens"
        else:
            return f""


    def loadPDF(self, inputFile : str) -> str :
        """
        Load text from PDF
        
        :param inputFile: PDF file name
        :type inputFile: str
        :return: Text from PDF
        :rtype: str
        """
        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()

        textCombined = ""
        for page in docs:
            pageContent = page.page_content
            if "stripWhiteSpace" in self.context and self.context["stripWhiteSpace"]:
                pageContent = pageContent.strip()
            if "convertToLower" in self.context and self.context["convertToLower"]:
                pageContent = pageContent.lower()
            if "convertToASCII" in self.context and self.context["convertToASCII"]:
                pageContent = anyascii(pageContent)
            if "singleSpaces" in self.context and self.context["singleSpaces"]:
                pageContent = " ".join(pageContent.split())
            textCombined += " "
            textCombined += pageContent
        return textCombined


    def getAbsPath(self, key) -> str:
        """
        absolute path value from relative path. Compatible with Django web app
        
        :param key: key in context dict
        :return: absolute path
        :rtype: str
        """        
        return Path(str(Path(__file__).parent.resolve()) + '/' + self.context[key]).resolve()


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
        with open(self.context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self.context, indent=2)
            jsonOut.write(formattedOut)


    def workerResult(self, msg : List[str]):
        """
        Logs result and updates status file

        Args:
            msg (str) - list of message strings 

        Returns:
            None
        """
        if msg:
            self.logger.info(msg)
            if 'results' not in self.context:
                self.context['results'] = []
            for oneMsg in msg:
                self.context['results'].append(oneMsg)
        with open(self.context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(self.context, indent=2)
            jsonOut.write(formattedOut)
