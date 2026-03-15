#
# base class for workflows
#
import sys
import logging
from logging import Logger
import json
from typing import List
from typing_extensions import Self
from pathlib import Path
import tomli


from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.google import GoogleModel

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.usage import RunUsage

from openai import AsyncOpenAI

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader

from anyascii import anyascii


# local
from common import GLOBALPROVIDER, PROVIDERS, COLLECTION, ConfigCollection, OpenFile


class WorkflowBase(BaseModel):
    """
    Base class for workflows
    """

    logger : Logger = Field(default = None, description="Application logger") 
    globalRAGDatapath : str = Field(default = "chromadb", description="Path to RAG database")
    globalRAGHNSWspace : str = Field(default = "cosine", description="Hierarchical Navigable Small World (HNSW) search algorithm similarity metric")
    globalProvider : str = Field(default = "", description="Global provider of LLM service") 
    embeddingLLM : str = Field(default = "", description="Embedding LLM") 
    embeddingURL : str = Field(default = "", description="Embedding LLM") 
    generalLLM : str = Field(default = "", strict=True, description="General LLM") 
    globalURL : str = Field(default = "", description="Global LLM service base URL") 
    globalAPIkey : str = Field(default = "", description="Global API Key") 
    chromaClient : ClientAPI = Field(default = None, description="ChromaDB Persistent Client") 
    embeddingFunction : Embedder = Field(default = None, description="ChromaDB embedding Function") 
    collections : dict[str, Collection] = Field(default = {}, description="dictionary of ChromaDB collections") 
    usage : RunUsage = Field(default = None, description="LLM usage object")
    stage : str = Field(default = "", description="Stage of workflow") 
    statusLog : List[str] = Field(default = [], description="Status log of workflow") 
    statusFileName : str = Field(default = "", description="Name of status log file") 
    resultsLog : List[str] = Field(default = [], description="Results log of workflow") 


    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def verify_configuration(self) -> Self:
        if self.globalProvider:
            if self.globalProvider not in PROVIDERS.keys():
                raise ValueError(f'Unknown LLM provider: {self.globalProvider}')
            providerInfo = PROVIDERS[self.globalProvider]
            if providerInfo["embed"] != self.embeddingURL:
                raise ValueError(f'LLM provider: {self.globalProvider} - Unknown Embedding URL: {self.embeddingURL}')
            if providerInfo["url"] != self.globalURL:
                raise ValueError(f'LLM provider: {self.globalProvider} - Unknown LLM API URL: {self.globalURL}')
            if self.generalLLM not in providerInfo["llm"]:
                raise ValueError(f'LLM provider: {self.globalProvider} - Unknown LLM: {self.generalLLM}')
        return self


    def configure(self, configCollection : ConfigCollection) :

        self.logger = logging.getLogger(configCollection["DISCLIsession_key"])

        self.globalProvider = configCollection["GLOBALllm_Provider"]
        self.embeddingLLM = configCollection["GLOBALllm_Embed"]
        self.embeddingURL = configCollection["GLOBALembedding_URL"]
        self.generalLLM = configCollection["GLOBALllm_Version"]
        self.globalURL = configCollection["GLOBALllm_URL"]
        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:
            self.globalAPIkey = configCollection['gemini_key']

        self.verify_configuration()

        self.embeddingFunction  = self.createEmbeddingFunction()
        self.chromaClient = self.openChromaClient()
        self.collections = {}
        if self.chromaClient:
            for coll in list(COLLECTION):
                self.collections[coll.value] = self.openOrCreateCollection(coll.value, True)
        self.usage = RunUsage()
        self.statusFileName = configCollection["statusFileName"]


    def openChromaClient(self) -> ClientAPI :
        """
        Opens ChromaDB client for vector database
        
        :return: ChromaDB persistent client or None
        :rtype: ClientAPI
        """
        try:
            chromaClient = chromadb.PersistentClient(
#                path=self.getAbsPath("GLOBALrag_datapath"),
                path=self.globalRAGDatapath,
                settings=Settings(anonymized_telemetry=False),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        except Exception as e:
            msg = f"Error: ChromaDB exception: {e}"
            self.workerError(msg)
            return None

        return chromaClient


    def createEmbeddingFunction(self) :
        """
        Create embedding function
        
        :return: embedding function object
        """

        if self.globalProvider == GLOBALPROVIDER.OLLAMA.value:
            
            return OllamaEmbeddingFunction(
                model_name=self.embeddingLLM,
                url=self.embeddingURL
            )

        if self.globalProvider == GLOBALPROVIDER.LMSTUDIO.value:

            model = OpenAIEmbeddingModel(
                self.embeddingLLM,
                provider=OpenAIProvider(
                    base_url=self.embeddingURL
                ),
            )
            return Embedder(model)

        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:

            model = OpenAIEmbeddingModel(
                self.embeddingLLM,
                provider=OpenAIProvider(
                    base_url=self.globalURL,
                    api_key=self.globalAPIkey
                ),
            )
            return Embedder(model)

        return None
    

    def createOpenAIChatModel(self) -> OpenAIChatModel: 
        """
        return OpenAIChatModel class instance or None
        
        :return: OpenAIChatModel class instance
        :rtype: OpenAIChatModel
        """

        if self.globalProvider == GLOBALPROVIDER.OLLAMA.value:
            return OpenAIChatModel(model_name=self.generalLLM,
                                   provider=OllamaProvider(base_url=self.globalURL))

        if self.globalProvider == GLOBALPROVIDER.LMSTUDIO.value:

            client = AsyncOpenAI(base_url=self.globalURL)
            return OpenAIChatModel(model_name=self.generalLLM, 
                                   provider=OpenAIProvider(openai_client=client))
        
        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:

            provider = GoogleProvider(api_key=self.globalAPIkey)
            return GoogleModel(self.geminiLLM , provider=provider)

        return None


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
                        metadata={ "hnsw:space": self.globalRAGHNSWspace }
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




#    def getAbsPath(self, key) -> str:
#        """
#        absolute path value from relative path. Compatible with Django web app
#        
#        :param key: key in context dict
#        :return: absolute path
#        :rtype: str
#        """        
#        return Path(str(Path(__file__).parent.resolve()) + '/' + self.context[key]).resolve()


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
            self.statusLog.append(msg)
        with open(self.statusFileName, "w") as jsonOut:
            formattedOut = json.dumps(self.statusLog, indent=2)
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
            self.statusLog.append(msg)
        with open(self.statusFileName, "w") as jsonOut:
            formattedOut = json.dumps(self.statusLog, indent=2)
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
            for oneMsg in msg:
                self.resultsLog.append(oneMsg)
        with open(self.statusFileName, "w") as jsonOut:
            formattedOut = json.dumps(self.resultsLog, indent=2)
            jsonOut.write(formattedOut)
