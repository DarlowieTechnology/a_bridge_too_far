#
# base class for workflows
#
import sys
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
from common import PROVIDERS, COLLECTION, OpenFile


class WorkflowBase(BaseModel):
    """
    Base class for workflows
    """

    context : dict = Field(..., description="Context dictionary", exclude=True) 
    logger : Logger = Field(..., description="Application logger", exclude=True) 
    globalProvider : str = Field(..., description="Global provider of LLM service", exclude=True) 
    embeddingLLM : str = Field(..., description="Embedding LLM", exclude=True) 
    embeddingURL : str = Field(..., description="Embedding LLM", exclude=True) 
    generalLLM : str = Field(..., strict=True, description="LM Studio LLM", exclude=True) 
    globalURL : str = Field(..., description="Global LLM service base URL", exclude=True) 
    chromaClient : ClientAPI = Field(..., description="ChromaDB Persistent Client", exclude=True) 
    embeddingFunction : Embedder = Field(..., description="ChromaDB embedding Function", exclude=True) 
    collections : dict[str, Collection] = Field(..., description="dictionary of ChromaDB collections", exclude=True) 
    usage : RunUsage = Field(..., description="LLM usage object", exclude=True)


    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def verify_configuration(self) -> Self:
        if self.globalProvider not in PROVIDERS.keys():
            raise ValueError(f'Unknown LLM provider: {self.globalProvider}')
        providerInfo = PROVIDERS[self.globalProvider]
        if providerInfo["embed"] != self.embeddingURL:
            raise ValueError(f'Unknown Embedding URL: {self.embeddingURL}')
        if providerInfo["url"] != self.globalURL:
            raise ValueError(f'Unknown LLM API URL: {self.globalURL}')
        if self.generalLLM not in providerInfo["llm"]:
            raise ValueError(f'Unknown LLM: {self.generalLLM}')
        return self


    def __init__(self, **kwargs):
#    def __init__(self, context : dict, logger : Logger, createCollection : bool):
        """
        Constructor for the base workflow object

        Args:
            context (dict) - context data for workflow
            logger (Logger) - created by caller (CLI or web app)
            createCollection (bool) - if True, create vector database
        """

        super().__init__(**kwargs)
#        super().__init__(context=context, logger=logger, globalProvider = "", embeddingLLM ="", generalLLM = "", embeddingURL = "", globalURL = "" )

        configName = 'default.toml'
        conf_dict = {}
        try:
            with open(configName, mode="rb") as fp:
                conf_dict = tomli.load(fp)
        except Exception as e:
            try:
                configName = "../" + configName
                with open(configName, mode="rb") as fp:
                    conf_dict = tomli.load(fp)
            except Exception as e:
                print(f"***ERROR: Cannot open config file {configName}, exception {e}")
                sys.exit("Program terminates")

        self.globalProvider = conf_dict["GLOBALllm_Provider"]
        self.embeddingLLM = conf_dict["GLOBALllm_Embed"]
        self.embeddingURL = conf_dict["GLOBALembedding_URL"]
        self.generalLLM = conf_dict["GLOBALllm_Version"]
        self.globalURL = conf_dict["GLOBALllm_Version"]

        self.context = context
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


    def createEmbeddingFunction(self) :
        """
        Create embedding function
        
        :return: embedding function object
        """

        if self.globalProvider == GLOBALPROVIDER.OLLAMA.value:
            
            return OllamaEmbeddingFunction(
                model_name=self.ollamaEmbed,
                url=self.globalURL + "/api/embeddings"
            )

        if self.globalProvider == GLOBALPROVIDER.LMSTUDIO.value:

            model = OpenAIEmbeddingModel(
                self.lmStudioEmbed,
                provider=OpenAIProvider(
                    base_url=self.globalURL,
                    api_key=self.config["LMSTUDIO_API_KEY"]
                ),
            )
            return Embedder(model)

        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:

            model = OpenAIEmbeddingModel(
                self.geminiEmbed,
                provider=OpenAIProvider(
                    base_url=self.globalURL,
                    api_key=self.config["gemini_key"]
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
            return OpenAIChatModel(model_name=self.ollamaLLM, 
                                   provider=OllamaProvider(base_url=self.globalURL))

        if self.globalProvider == GLOBALPROVIDER.LMSTUDIO.value:

            client = AsyncOpenAI(base_url=self.globalURL)
            return OpenAIChatModel(model_name=self.lmStudioLLM, 
                                   provider=OpenAIProvider(openai_client=client))
        
        if self.globalProvider == GLOBALPROVIDER.GEMINI.value:

            provider = GoogleProvider(api_key=self.config["gemini_key"])
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
#                name=collectionName,
#                embedding_function=self.embeddingFunction
                name=collectionName
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
