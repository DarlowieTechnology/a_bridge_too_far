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

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.models.google import GoogleModel

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.usage import RunUsage

from openai import Model, AsyncOpenAI

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader

from anyascii import anyascii


# local
from common import GLOBALPROVIDER, PROVIDERS, LLMNAMES, OPENAIAPI, COLLECTION, ConfigCollection, OpenFile


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

        # manually call model validator
        self.verify_configuration()

        # delay RAG components till vectorize action
#        self.embeddingFunction  = self.createEmbeddingFunction()
#        self.chromaClient = self.openChromaClient()
#        if self.chromaClient:
#            for coll in list(COLLECTION):
#                self.collections[coll.value] = self.openOrCreateCollection(coll.value, True)

        self.usage = RunUsage()
        self.statusFileName = configCollection["statusFileName"]


    def initRAGcomponents(self) -> bool :
        """
        Init RAG components on demand.
        """
        self.embeddingFunction  = self.createEmbeddingFunction()
        if not self.embeddingFunction:
            return False
        self.chromaClient = self.openChromaClient()
        if self.chromaClient:
            for coll in list(COLLECTION):
                self.collections[coll.value] = self.openOrCreateCollection(coll.value, True)
        return True


    def getModel(self, modelType : OPENAIAPI) -> Model :
        """
        Return Pydantic OpenAI Model instance
        """

        if modelType == OPENAIAPI.CHAT:
            OpenAIclient = AsyncOpenAI(max_retries=3, base_url=self.globalURL)
            OpenAIprovider = OpenAIProvider(openai_client=OpenAIclient)
            model = OpenAIChatModel(model_name=self.generalLLM, 
                                provider=OpenAIprovider)
            return model

        if modelType == OPENAIAPI.RESPONSES:
            OpenAIclient = AsyncOpenAI(max_retries=3, base_url=self.globalURL)
            OpenAIprovider = OpenAIProvider(openai_client=OpenAIclient)
            model = OpenAIResponsesModel(model_name=self.generalLLM, 
                                provider=OpenAIprovider)
            return model

        if modelType == OPENAIAPI.CHATOLLAMA:
            ollamaProvider=OllamaProvider(base_url=self.globalURL)
            model = OpenAIChatModel(model_name=self.generalLLM, 
                                    provider=ollamaProvider)
            return model
        
        if modelType == OPENAIAPI.CHATGEMINI:
            provider = GoogleProvider(api_key=self.globalAPIkey)
            model =  GoogleModel(self.generalLLM , provider=provider)
            return model

        msg = f"Error: unknown OpenAI model type."
        self.workerError(msg)
        return None


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

            return OllamaEmbeddingFunction(
                model_name=self.embeddingLLM,
                url=self.embeddingURL
            )

        msg = f"Error: Unknown provider. Cannot create embedding function."
        self.workerError(msg)
        return None
    

    def createOpenAIModel(self) -> Model: 
        """
        return OpenAI Model class instance or None
        
        :return: OpenAI Model class instance
        :rtype: OpenAI Model
        """

        if self.globalProvider == GLOBALPROVIDER.OLLAMA.value:

            # Ollama uses OpenAI Chat API
            return self.getModel(OPENAIAPI.CHATOLLAMA)
            
        if self.globalProvider == GLOBALPROVIDER.LMSTUDIO.value:

            if (self.generalLLM == LLMNAMES.GPTOSS120B.value) or (self.generalLLM == LLMNAMES.GPTOSS20B.value):

                # LM Studio uses OpenAI Responses API for gpt-oss models
                return self.getModel(OPENAIAPI.RESPONSES)

            if (self.generalLLM == LLMNAMES.LLAMA2.value) or (self.generalLLM == LLMNAMES.LLAMA33.value):

                # LM Studio uses OpenAI Chat API for LLama models
                return self.getModel(OPENAIAPI.CHAT)

        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:

            # Google Gemini uses OpenAI Chat API
            return self.getModel(OPENAIAPI.CHATGEMINI)

        msg = f"Error: Unknown combination of provider and model name"
        self.workerError(msg)
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
