#
# base class for workflows
#
import sys
import logging
from logging import Logger
import json
from typing import List, Union, Dict, Any
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
from common import GLOBALPROVIDER, LLMNAMES, OPENAIAPI, COLLECTION, ConfigCollection, DebugUtils


class WorkflowBase(BaseModel):
    """
    Base class for workflows
    """

    logger : Logger = Field(default = None, description="Application logger", exclude=True)
    chromaClient : ClientAPI = Field(default = None, description="ChromaDB Persistent Client", exclude=True)
    embeddingFunction : OllamaEmbeddingFunction = Field(default = None, description="ChromaDB embedding Function", exclude=True)
    collections : dict[str, Collection] = Field(default = {}, description="dictionary of ChromaDB collections", exclude=True)
    usage : RunUsage = Field(default = None, description="LLM usage object", exclude=True)
    statusLog : List[str] = Field(default = [], description="Status log of workflow", exclude=True)
    stats : dict[str, dict[str, Union[int, str]]] = Field(default = {}, description="Run statistics", exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude=True)


    @model_validator(mode='after')
    def verify_configuration(self) -> Self:
        return self


    def configure(self, configCollection : ConfigCollection):

        logging.basicConfig(stream=sys.stdout, level=configCollection["GLOBALloggerLevel"])
        self.usage = RunUsage()


    def needsUpdate(self, updatedDiscoveryWorkflow : Dict[str, Any]) -> bool:
        toUpdate = False
        for key in updatedDiscoveryWorkflow.keys():
            if hasattr(self, key):
                toUpdate = toUpdate or (self.__dict__[key] != updatedDiscoveryWorkflow[key])
        return toUpdate


    def updateStats(self, topKey : str, keyValList : List[tuple[str, int]]) :
        """
        Update internal statistics. Attempt to update first, create key second.
        
        :param topKey:  key for stat record
        :type topKey: str
        :param keyValList:  list of stats tuples (key-val)
        :type keyValList: List[tuple[str, int]]
        :return: None
        :rtype: None
        """

        try:
            statsForKey = self.stats[topKey]
        except:
            self.stats[topKey] = {}
            statsForKey = self.stats[topKey]

        for key, value in keyValList:
            try:
                prevVal = statsForKey[key]
                if type(prevVal) == str:
                    statsForKey[key] = value
                else:
                    statsForKey[key] = prevVal + value
            except Exception:
                statsForKey[key] = value


    def removeStats(self, topKey : str, removeKey : str) :
        """
        Remove key from internal statistics
        
        :param topKey:  key for stat record
        :type topKey: str
        :param removeKey:  key to remove
        :type removeKey: str
        :return: None
        :rtype: None
        """

        try:
            statsForKey = self.stats[topKey]
            statsForKey.pop(removeKey)
        except:
            pass


    def showStats(self, topKey : str, showKey : str, label : str) -> str:
        """
        Update internal statistics. Attempt to update first, create key second.
        
        :param topKey:  key for stat record
        :type topKey: str
        :param showKey:  key to show
        :type showKey: str
        :param label:  label to show
        :type label: str
        :return: String to show
        :rtype: str
        """

        try:
            statsForKey = self.stats[topKey]
            showValue = statsForKey[showKey]
        except:
            return ""
        return f"{label} : {showValue}"



    def initRAGcomponents(self) -> bool :
        """
        Init RAG components on demand. Avoid repeated initialization.
        """
        if not self.embeddingFunction:
            self.embeddingFunction  = self.createEmbeddingFunction()
        if not self.embeddingFunction:
            return False
        if not self.chromaClient:
            self.chromaClient = self.openChromaClient()
            if not self.chromaClient:
                return False
        if self.chromaClient:
            for coll in list(COLLECTION):
                if coll.value in self.collections.keys():
                    if not self.collections[coll.value]:
                        self.collections[coll.value] = self.openOrCreateCollection(coll.value, True)
                    if not self.collections[coll.value]:
                        return False
                else:
                    self.collections[coll.value] = self.openOrCreateCollection(coll.value, True)
                    if not self.collections[coll.value]:
                        return False
        return True


    def getChromaCollection(self, name: str) -> Collection:
        """
        Returns Chroma collection by name or None
        """
        if name in self.collections.keys():
            return self.collections[name]
        return None


    def getModel(self, modelType : OPENAIAPI) -> Model :
        """
        Return Pydantic OpenAI Model instance
        """

        if modelType == OPENAIAPI.CHAT:

#            print(f"globalURL: {self.globalURL}  general LLM: {self.generalLLM}")

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
                path=self.ragDatapath,
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
        TODO: LM Studio uses its own API to perform embedding 
        
        :return: embedding function object
        """

        ollamaEmbeddingFunction = OllamaEmbeddingFunction(
            model_name=self.embeddingLLM,
            url=self.embeddingURL
        )
        return ollamaEmbeddingFunction


        if self.globalProvider == GLOBALPROVIDER.OLLAMA.value:
            
            ollamaEmbeddingFunction = OllamaEmbeddingFunction(
                model_name=self.embeddingLLM,
                url=self.embeddingURL
            )
            return ollamaEmbeddingFunction

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

            if (self.generalLLM == LLMNAMES.GPTOSS120BLMSTUDIO.value) or (self.generalLLM == LLMNAMES.GPTOSS20BLMSTUDIO.value):

                # LM Studio uses OpenAI Responses API for gpt-oss models
                return self.getModel(OPENAIAPI.RESPONSES)
#                return self.getModel(OPENAIAPI.CHAT)
            
            if (self.generalLLM == LLMNAMES.LLAMA3370BLMSTUDIO.value) or (self.generalLLM == LLMNAMES.GEMMA4LMSTUDIO.value):

                # LM Studio uses OpenAI Chat API for LLama models and Google Gemma models
#                print("Creating OPENAIAPI.CHAT")
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
                embedding_function=self.embeddingFunction,
                name=collectionName
            )
            msg = f"Opened collections {collectionName} with {chromaCollection.count()} documents."
            self.workerSnapshot(msg)
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


    def totalUsageFormat(self, insertHTML : bool) -> str:
        """
        Return current usage formatted as string
        
        Args:
            InsertHTML (bool) - insert bold tags for HTML page
        :return: usage as string
        :rtype: str
        """
        requestLabel = 'requests' if self.usage.requests > 1 else 'request'
        if insertHTML:
            return f"<b>{self.usage.requests}</b> {requestLabel}, <b>{self.usage.input_tokens}>/b> input tokens, <b>{self.usage.output_tokens}</b> output tokens"
        else:
            return f"{self.usage.requests} {requestLabel}, {self.usage.input_tokens} input tokens, {self.usage.output_tokens} output tokens"


    def usageFormat(self, usage : RunUsage, insertHTML : bool) -> str:
        """
        Return usage formatted as string

        Args:
            usage (runUsage) - usage object
            InsertHTML (bool) - insert bold tags for HTML page
        
        :return: usage as string
        :rtype: str
        """
        if usage:
            requestLabel = 'requests' if usage.requests > 1 else 'request'
            if insertHTML:
                return f"<b>{usage.requests}</b> {requestLabel}, <b>{usage.input_tokens}</b>, input tokens <b>{usage.output_tokens}</b> output tokens"
            else:
                return f"{usage.requests} {requestLabel}, {usage.input_tokens} input tokens, {usage.output_tokens} output tokens"
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
