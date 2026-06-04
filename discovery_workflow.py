#
# Discovery workflow class used by Django app and command line
#
from typing import List, Dict, Any
from typing_extensions import Self
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4
import hashlib
import re
from pprint import pprint

from pydantic import BaseModel, ConfigDict, Field, model_validator
import pydantic_ai
from pydantic_ai import Agent, AgentRunResult

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.google import GoogleModel

from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.google import GoogleProvider

from pydantic_ai.usage import RunUsage

import pandas as pd

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

import Stemmer
import bm25s
import spacy
from spacy import Language


from anyascii import anyascii

# local
from common import PROVIDERS, GLOBALPROVIDER, COLLECTION, TOKENIZERTYPES, CommonHelper, ConfigCollection, MatchingChunks, AllTopicMatches, ChunkInfo, OpenFile
from resultsQueryClasses import SEARCH, OneQueryChunkResult, OneChunkQueryResultList, IdentifierQueryResults, RRFScores, AllChunkQueryResults, CollectionChunkQueryResults
from queryService import QueryService
from workflowbase import WorkflowBase 


acceptedMimeTypes = [
    "text/css",
    "text/csv",
    "text/html",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/plain"
]


class DiscoveryWorkflow(WorkflowBase):

    GLOBALllm_Provider : str = Field(default = "", description="Global provider of LLM service")
    GLOBALllm_Version : str = Field(default = "", strict=True, description="General LLM")
    GLOBALllm_URL : str = Field(default = "", description="Global LLM service base URL")
    GLOBALllm_Embed : str = Field(default = "", description="Embedding LLM")
    GLOBALembedding_URL : str = Field(default = "", description="Embedding LLM")
    gemini_key : str = Field(default = "", description="Global Google Gemini API Key")

    # app specific configuration
    statusLog : List[str] = Field(default = [], description="Status log of workflow")
    statusFileName : str = Field(default = "DISCOVERYLOG", description="Name of status log file")
    ragDatapath : str = Field(default = "chromadb", description="Path to RAG database")
    documentFolder : str = Field(default = "", description="Source document folder")
    dataFolder : str = Field(default = "", description="Intermediate data folder")
    bm25IndexFolder : str = Field(default = "", description="bm25 index folder")

    source : List[str] = Field(default = [], description="List of source documents")
    fileExtensions : List[str] = Field(default = ["*.txt", "*.pdf", "*.json"], description="List of source file allowed file name extensions")
    chunkSize : str = Field(default = "256", description="Chunk size for source documents")
    chunkOverlap : str = Field(default = "48", description="Chunk overlap for source documents")

    taskId : str = Field(default = "", description="UUID4 of the task")
    inWorkflow : bool = Field(default = False, description="Flag is True when Workflow is active")

    # text processing flags
    stripWhiteSpace : bool = Field(default = True, description="Strip excessive whitespace characters from source text")
    convertToLower : bool = Field(default = True, description="Covert all characters in source text to lowercase")
    convertToASCII : bool = Field(default = True, description="Covert all characters in source text to ASCII")
    singleSpaces : bool = Field(default = True, description="Replace multiple space characters with single space in source text")

    # workflow actions
    loadDocument : bool = Field(default = True, description="Load text from source documents")
    parseChunks : bool = Field(default = True, description="Create chunks out of text")
    makeRawVector : bool = Field(default = True, description="Vectorize chunks in vector table")
    bm25Process : bool = Field(default = True, description="Create bm25 index")
    search : bool = Field(default = True, description="Search chunks for known topics")
    clear : bool = Field(default = False, description="Clear Intermediate data")

    # search configuration
    query : List[str] = Field(default = [], description="List of known topics to use in queries")
    searchSemanticOriginal : bool = Field(default = True, description="Perform original semantic query")
    searchBM25sOriginal : bool = Field(default = True, description="Perform original bm25s query")
    searchSemanticMulti : bool = Field(default = True, description="Perform semantic query on multi transform")
    searchBM25sMulti : bool = Field(default = True, description="Perform bm25s query on multi transform")
    searchSemanticRewrite : bool = Field(default = True, description="Perform semantic query on rewrite transform")
    searchBM25sRewrite : bool = Field(default = True, description="Perform bm25s query on rewrite transform")
    searchSemanticHyDE : bool = Field(default = True, description="Perform semantic query on HyDE transform")
    searchBM25sHyDE : bool = Field(default = True, description="Perform bm25s query on HyDE transform")

    # retrieval configuration
    semanticRetrieveNumber : str = Field(default = "50", description="Number of items retrieved with semantic query")
    semanticMaxCutItemDistance: str  = Field(default = "1.0", description="Maximum distance in semantic search")
    bm25sRetrieveNumber : str = Field(default = "50", description="Number of items retrieved with bm25s query")
    bm25sMinCutOffScore : str = Field(default = "0.0", description="Minimum bm25s score cut off")
    rrfCutOffValue : str = Field(default = "0.0", description="Reciprocal Rank Fusion (RRF) value cut off")
    rrfOutlierZScoreThreshold : float = Field(default = 3.0, description="Threshold for outlier z-score")
    rrfOutlierIQRCoefficient : float = Field(default = 1.5, description="Interquartile Range (IQR) upper fence coefficient")
    outputNumber : str = Field(default = "50", description="Maximum number of items to return")

    outputFileName : str = Field(default = "../testdata/discoverydocuments/discoverydata/DISCOVERY.results.json", description="File name for results")

    @model_validator(mode='after')
    def verify_configuration(self) -> Self:

        if self.GLOBALllm_Provider:
            if self.GLOBALllm_Provider not in PROVIDERS.keys():
                raise ValueError(f'Unknown LLM provider: {self.GLOBALllm_Provider}')
            providerInfo = PROVIDERS[self.GLOBALllm_Provider]
            if providerInfo["embed"] != self.GLOBALembedding_URL:
                raise ValueError(f'LLM provider: {self.GLOBALllm_Provider} - Unknown Embedding URL: {self.GLOBALembedding_URL}')
            if providerInfo["url"] != self.GLOBALllm_URL:
                raise ValueError(f'LLM provider: {self.GLOBALllm_Provider} - Unknown LLM API URL: {self.GLOBALllm_URL}')
            if self.GLOBALllm_Version not in providerInfo["llm"]:
                raise ValueError(f'LLM provider: {self.GLOBALllm_Provider} - Unknown LLM: {self.GLOBALllm_Version}')

        # verify access to Source document folder
        if not Path(self.documentFolder).is_dir:
            raise ValueError(f'Source document folder is invalid')
        # verify access to Intermediate data folder
        if not Path(self.dataFolder).is_dir:
            raise ValueError(f'Intermediate data folder is invalid')
        # verify access to bm25 index folder
        if not Path(self.dataFolder + self.bm25IndexFolder).is_dir:
            raise ValueError(f'bm25 index folder is invalid')
        # verify formatting of file extension list
        for fileExt in self.fileExtensions:
            extPattern = r"\*\.[A-Za-z0-9]*$"
            if not re.match(extPattern, fileExt):
                raise ValueError(f'File extension pattern is invalid')
        if not int(self.chunkSize) in range(128, 513):
            raise ValueError(f'Chunk size is invalid')
        if not int(self.chunkOverlap) in range(0, 65):
            raise ValueError(f'Chunk overlap is invalid')
        
        if not int(self.semanticRetrieveNumber) in range(0, 2049):
            raise ValueError(f'Number of semantic search items is invalid')
        if not float(self.semanticMaxCutItemDistance) >= 0 and float(self.semanticMaxCutItemDistance) <= 1.0:
            raise ValueError(f'Maximum distance of semantic search items is invalid')
        if not int(self.bm25sRetrieveNumber) in range(0, 2049):
            raise ValueError(f'Number of bm25s search items is invalid')
        if not float(self.bm25sMinCutOffScore) >= 0:
            raise ValueError(f'bm25s score cut off value is invalid')
        if not (float(self.rrfCutOffValue) >= 0 and float(self.rrfCutOffValue) <= 1.0):
            raise ValueError(f'Reciprocal Rank Fusion (RRF) cut off value is invalid')
        if not (self.rrfOutlierZScoreThreshold >= 0):
            raise ValueError(f'Z Score threshold for outliers is invalid')
        if not (self.rrfOutlierIQRCoefficient >= 0):
            raise ValueError(f'IQR Coefficient for outliers is invalid')
        if not int(self.outputNumber) in range(1, 2049):
            raise ValueError(f'output number is invalid')

        if not Path(self.outputFileName).is_file:
            raise ValueError(f'Output file name is invalid')

        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        self.GLOBALllm_Provider = configCollection["GLOBALllm_Provider"]
        self.GLOBALllm_Embed = configCollection["GLOBALllm_Embed"]
        self.GLOBALembedding_URL = configCollection["GLOBALembedding_URL"]
        self.GLOBALllm_Version = configCollection["GLOBALllm_Version"]
        self.GLOBALllm_URL = configCollection["GLOBALllm_URL"]
        if self.GLOBALllm_Provider == GLOBALPROVIDER.GEMINI.value:
            self.gemini_key = configCollection['gemini_key']

        if configCollection.keyExists("statusFileName"):
            self.statusFileName = configCollection["statusFileName"]

        if configCollection.keyExists("ragDatapath"):
            # WEB update, CLI advanced settings
            self.ragDatapath = configCollection["ragDatapath"]
        else:
            # CLI and WEB init
            self.ragDatapath = configCollection["DISCOVRAGFolder"]

        if configCollection.keyExists("documentFolder"):
            # WEB update, CLI advanced settings
            self.documentFolder = configCollection["documentFolder"]
        else:
            # CLI and WEB init
            self.documentFolder = configCollection["DISCOVdocumentFolder"]

        if configCollection.keyExists("dataFolder"):
            # WEB update, CLI advanced settings
            self.dataFolder = configCollection["dataFolder"]
        else:
            # CLI and WEB init
            self.dataFolder = configCollection["DISCOVdataFolder"]

        if configCollection.keyExists("bm25IndexFolder"):
            # WEB update, CLI advanced settings
            self.bm25IndexFolder = configCollection["bm25IndexFolder"]
        else:
            # CLI and WEB init
            self.bm25IndexFolder = configCollection["DISCOVbm25IndexFolder"]

        # make bm25s index folder if does not exist
        Path(self.bm25IndexFolder).mkdir(parents=True, exist_ok=True)

        # make data folder if does not exist
        Path(self.dataFolder).mkdir(parents=True, exist_ok=True)


        # workflow actions
        if configCollection.keyExists("loadDocument"): 
            self.loadDocument = configCollection["loadDocument"]
        if configCollection.keyExists("parseChunks"): 
            self.parseChunks = configCollection["parseChunks"]
        if configCollection.keyExists("makeRawVector"): 
            self.makeRawVector = configCollection["makeRawVector"]
        if configCollection.keyExists("bm25Process"): 
            self.bm25Process = configCollection["bm25Process"]
        if configCollection.keyExists("search"): 
            self.search = configCollection["search"]
        if configCollection.keyExists("clear"): 
            self.clear = configCollection["clear"]

        if configCollection.keyExists("stripWhiteSpace"): 
            self.stripWhiteSpace = configCollection["stripWhiteSpace"]
        if configCollection.keyExists("convertToLower"): 
            self.convertToLower = configCollection["convertToLower"]
        if configCollection.keyExists("convertToASCII"): 
            self.convertToASCII = configCollection["convertToASCII"]
        if configCollection.keyExists("singleSpaces"): 
            self.singleSpaces = configCollection["singleSpaces"]

        if configCollection.keyExists("source"):
            fileList = configCollection["source"]
            if len(fileList):
                self.source = []
                for fileName in fileList:
                    res, err = OpenFile.open(fileName, False)
                    if not res:
                        fileName = self.documentFolder + fileName
                        res, err = OpenFile.open(fileName, False)
                    if res:
                        self.source.append(fileName)
            else:
                self.source = self.formFileList()    
        else:
            self.source = self.formFileList()

        if configCollection.keyExists("fileExtensions"):
            self.fileExtensions = configCollection["fileExtensions"]
        if configCollection.keyExists("chunkSize"):
            self.chunkSize = configCollection["chunkSize"]
        if configCollection.keyExists("chunkOverlap"):
            self.chunkOverlap = configCollection["chunkOverlap"]

        # search configuration
        if configCollection.keyExists("query"):
            self.query = configCollection["query"]

        if configCollection.keyExists("searchSemanticOriginal"):
            self.searchSemanticOriginal = configCollection["searchSemanticOriginal"]
        if configCollection.keyExists("searchBM25sOriginal"):
            self.searchBM25sOriginal = configCollection["searchBM25sOriginal"]
        if configCollection.keyExists("searchSemanticMulti"): 
            self.searchSemanticMulti = configCollection["searchSemanticMulti"]
        if configCollection.keyExists("searchBM25sMulti"):
            self.searchBM25sMulti = configCollection["searchBM25sMulti"]
        if configCollection.keyExists("searchSemanticRewrite"): 
            self.searchSemanticRewrite = configCollection["searchSemanticRewrite"]
        if configCollection.keyExists("searchBM25sRewrite"): 
            self.searchBM25sRewrite = configCollection["searchBM25sRewrite"]
        if configCollection.keyExists("searchSemanticHyDE"):
            self.searchSemanticHyDE = configCollection["searchSemanticHyDE"]
        if configCollection.keyExists("searchBM25sHyDE"):
            self.searchBM25sHyDE = configCollection["searchBM25sHyDE"]

        # retrieval configuration
        if configCollection.keyExists("semanticRetrieveNumber"): 
            self.semanticRetrieveNumber = configCollection["semanticRetrieveNumber"]
        if configCollection.keyExists("semanticMaxCutItemDistance"):
            self.semanticMaxCutItemDistance = configCollection["semanticMaxCutItemDistance"]
        if configCollection.keyExists("bm25sRetrieveNumber"):
            self.bm25sRetrieveNumber = configCollection["bm25sRetrieveNumber"]
        if configCollection.keyExists("bm25sMinCutOffScore"):
            self.bm25sMinCutOffScore = configCollection["bm25sMinCutOffScore"]
        if configCollection.keyExists("rrfCutOffValue"):
            self.rrfCutOffValue = configCollection["rrfCutOffValue"]
        if configCollection.keyExists("rrfOutlierZScoreThreshold"):
            self.rrfOutlierZScoreThreshold = configCollection["rrfOutlierZScoreThreshold"]
        if configCollection.keyExists("rrfOutlierIQRCoefficient"):
            self.rrfOutlierIQRCoefficient = configCollection["rrfOutlierIQRCoefficient"]
        if configCollection.keyExists("outputNumber"):
            self.outputNumber = configCollection["outputNumber"]

        self.stats = {}

        if configCollection.keyExists("outputFileName"):
            self.outputFileName = configCollection["outputFileName"]

        # manually call model validator
        self.verify_configuration()


    def processText(self, textIn : str) -> str:
        """
        Process text per flags (strip, lower, convert to ASCII, single space)
        
        :param textIn: text to process
        :type textIn: str
        :return: processed text
        :rtype: str
        """
        if self.stripWhiteSpace:
            textIn = textIn.strip()
        if self.convertToLower:
            textIn = textIn.lower()
        if self.convertToASCII:
            textIn = anyascii(textIn)
        if self.singleSpaces:
            textIn = " ".join(textIn.split())
        return textIn


    def loadPDFPyPDFLoader(self, inputFile : str) -> str|None :
        """
        Load text from PDF using PyPDFLoader
        
        :param inputFile: PDF file name
        :type inputFile: str
        :return: Text from PDF
        :rtype: str
        """
        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        try:
            docs = loader.load()
        except Exception as e:
            self.logMessage(f"Exception: {e}")
            return None       

        textCombined = ""
        for page in docs:
            pageContent = page.page_content
            pageContent = self.processText(pageContent)
            textCombined += " "
            textCombined += pageContent
        return textCombined


    def loadPDFpymupdf4llm(self, inputFile : str) -> str|None :
        """
        Load text from PDF using pymupdf4llm
        
        :param inputFile: PDF file name
        :type inputFile: str
        :return: Text from PDF
        :rtype: str
        """

        try:
            docs = pymupdf4llm.to_text(inputFile)
        except Exception as e:
            self.logMessage(f"Exception: {e}")
            return None       
        
        docs = self.processText(docs)
        return docs


    def loadText(self, inputFile : str) -> str :
        """
        Load text from text-type file
        
        :param inputFile: file name
        :type inputFile: str
        :return: Text
        :rtype: str
        """

        with open(inputFile, "r" , encoding="utf-8", errors="ignore") as txtIn:
            docs = txtIn.read()

        docs = self.processText(docs)
        return docs


    def loadJSON(self, inputFile : str) -> str :
        """
        Load text from JSON file via DataFrame
        
        :param inputFile: file name
        :type inputFile: str
        :return: Text
        :rtype: str
        """

        with open(inputFile, "r" , encoding="utf-8", errors="ignore") as JsonIn:
            docs = json.load(JsonIn)

        dataFrame = pd.DataFrame([docs])

        strOut = dataFrame.to_string(header = False, justify = "left")

        return strOut


    def formFileList(self) -> List[str]:
        """
        Return list of files for processing
        
        :return: list of files
        :rtype: List[str]
        """

        completeFileList = []
        for globName in self.fileExtensions:
            result, fileListOrError = OpenFile.readListOfFileNames(self.documentFolder, globName)
            if result:
                completeFileList = completeFileList + fileListOrError
        return completeFileList


    def compressText(self, textIn : str, nlp : Language) -> str:
        """
        Perform Telegraphic Semantic Compression (TSC) on the query for semantic search. 
        Ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/.
        Get english dictionary: python -m spacy download en_core_web_sm.

        :param textIn:  original chunk of text 
        :type textIn: str
        :return: compressed text
        :rtype: str
        """

        # Parts of speech to remove (predictable grammar)
        REMOVE_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}

        # Optional low-information words to remove
        REMOVE_LIKE = {"like", "just", "really", "basically", "literally", "have"}

        doc = nlp(textIn)

        chunks = []

        for sent in doc.sents:

            words = [
                token.lemma_
                for token in sent
                if (
                    token.pos_ not in REMOVE_POS
                    and token.text.lower() not in REMOVE_LIKE
                    and not token.is_punct
                )
            ]
            if words:
                chunks.append(" ".join(words))
           

        outText =  " ".join(chunks)

        return outText


    def outputRRFInfo(self, rrfScores : RRFScores, onlyOutliers : bool) -> List[str]:
        """
        Output RRF scores. Stop at RRF cut off value. 

        :param rrfScores:  RRFScores object
        :type rrfScores:  RRFScores
        :return: list of output strings
        :rtype: List[str]
        """

        outStrings = []
        for ident in rrfScores.scoresDict.keys():
            identifierQueryResults = rrfScores.scoresDict[ident]
            if identifierQueryResults.rrfRank < float(self.rrfCutOffValue):
                break
            if onlyOutliers:
                if identifierQueryResults.outlierIQR or identifierQueryResults.outlierZScore:
                    print(f"{identifierQueryResults.model_dump(mode = 'python')},")
#            outStrings.append((f"{identifierQueryResults.model_dump(mode = 'python')},"))

        return outStrings


    def loadDocumentPhase(self, inputFileName : str, dataFolder : str, outputFileName : str) -> int: 
        """
        Load document from one of various formats and store it as plain text

        :param inputFileName:  input file name  
        :type inputFileName:  str
        :param dataFolder:  temp data folder name
        :type dataFolder:  str
        :param outputFileName:  output file name
        :type outputFileName:  str
        :return: Length of extracted text 
        :rtype: int
        """
        
        self.logMessage(f"Load: {inputFileName}")
        mime_type, encoding = mimetypes.guess_type(inputFileName)
        if mime_type not in acceptedMimeTypes:
            self.logMessage(f"Error: {inputFileName} File type not supported: {mime_type}")
            self.updateStats(topKey = "Load", keyValList = [("Files", 1), ("Unknown MIME type", 1)])
            return 0

        if mime_type == "application/pdf":
            textCombined = self.loadPDFPyPDFLoader(inputFileName)
            if not textCombined:
                textCombined = self.loadPDFpymupdf4llm(inputFileName)
                self.updateStats(topKey = "Load", keyValList = [("pymupdf4llm", 1)])
            else:
                self.updateStats(topKey = "Load", keyValList = [("PyPDFLoader", 1)])
            if textCombined:
                self.updateStats(topKey = "Load", keyValList = [("Files", 1), ("Length", len(textCombined)), ("Type PDF", 1), ("Type PDF Total Length", len(textCombined))])

        if mime_type in ["application/json"]:
            textCombined = self.loadJSON(inputFileName)
            self.updateStats(topKey = "Load", keyValList = [("Files", 1), ("Length", len(textCombined)), ("Type JSON", 1), ("Type JSON Total Length", len(textCombined))])

        if mime_type in ["text/css", "text/csv", "text/html", "text/markdown", "text/plain"]:
            textCombined = self.loadText(inputFileName)
            self.updateStats(topKey = "Load", keyValList = [("Files", 1), ("Length", len(textCombined)), ("Type Other Text", 1), ("Other Text Total Length", len(textCombined))])

        fullOutputFileName = dataFolder + outputFileName
        if textCombined:
            with open(fullOutputFileName, "w" , encoding="utf-8", errors="ignore") as rawOut:
                rawOut.write(textCombined)
            return len(textCombined)
        return 0


    def loadDocumentPhaseAllFiles(self, inputFileList : List[str]) : 
        """
        For all files load document from one of various formats and store it as plain text

        :param inputFileList: source files to add
        :type List[str]
        :return: 
        :rtype: 
        """
        for inputFileName in inputFileList:
            intermediateDataFolder = self.dataFolder + Path(inputFileName).name + "-data/"
            Path(intermediateDataFolder).mkdir(parents=True, exist_ok=True)            
            self.loadDocumentPhase(inputFileName, intermediateDataFolder, "raw.txt")


    def parseChunksPhase(self, docs : str) -> List[str] :
        """
        split text into chunks using text splitter object
        
        :param docs: document
        :type docs: str
        :return: chunk list
        :rtype: List[str]
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(self.chunkSize), chunk_overlap=int(self.chunkOverlap))
        texts = text_splitter.split_text(docs)

        self.updateStats(topKey = "Chunking", keyValList = [("Chunks", len(texts))])
        return texts


    def parseChunksPhaseAllFiles(self, inputFileList : List[str]) :
        """
        for all files split text into chunks using text splitter object
        
        :param inputFileList: source files to add
        :type List[str]
        :return: 
        :rtype: 
        """
        for inputFileName in inputFileList:

            # read raw text into string
            rawTextFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.txt"
            result, fileContentOrError = OpenFile.open(filePath = rawTextFileName, readContent = True)
            if not result:
                # record error and attempt to process next file
                self.logMessage(f"parseChunks: {fileContentOrError} - perform '--load' phase first")
            else:
                self.logMessage(f"parseChunks: {inputFileName}")
                chunksList = self.parseChunksPhase(fileContentOrError)
                chunkListFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.chunks.txt"

                with open(chunkListFileName, "w" , encoding="utf-8", errors="ignore") as jsonOut:
                    jsonOut.writelines(json.dumps(chunksList, indent=2))



    def makeRawVectorPhase(self, doc : List[str], inputFileName : str) -> tuple[int, int]:
        """
        Add all chunks to raw vector database. 
        Chunk is rejected if sha256 digest and chunk sequence in the document are the same as recorded previously
        ChromaDB default embedding model is used
        
        :param doc: chunks to add
        :type List[str]
        :param inputFileName: source file name
        :type str
        :return: Tuple of accepted and rejected chunks
        :rtype: tuple[int, int]
        """

        accepted = 0
        rejected = 0

        if not self.initRAGcomponents():
            return accepted, rejected

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return accepted, rejected

        runId = str(uuid4())
        chunkId = -1
        ids : list[str] = []
        docs : list[str] = []
        docMetadata : list[str] = []
        embeddings = []

        for chunk in doc:
            chunkId += 1
            hashFunc = hashlib.sha256()
            hashFunc.update(chunk.encode('utf-8'))
            recordHash = hashFunc.hexdigest() + "," + str(chunkId) + "," + str(inputFileName)

            queryResult = chromaCollection.get(ids=[recordHash])
            if (len(queryResult["ids"])) :
                rejected += 1
                continue
            else:
                ids.append(recordHash)
                docs.append(chunk)
                metadataDict = {}
                metadataDict["document"] = str(inputFileName)
                metadataDict["runid"] = runId
                metadataDict["chunkid"] = str(chunkId)
                docMetadata.append( metadataDict )
                try:
                    embedding = self.embeddingFunction([chunk])
                except Exception as e:
                    self.logMessage(f"makeRawVector: {e}")
                    return -1, -1            
                embeddings.append(embedding[0])
                accepted += 1

        if len(ids):
            chromaCollection.add(
                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=docMetadata
            )

        return accepted, rejected


    def makeRawVectorPhaseAllFiles(self, inputFileList : List[str]) -> tuple[int, int]:
        """
        For all files add chunks to raw vector database. 
        
        :param inputFileList: source files to add
        :type List[str]
        :return: Tuple of accepted and rejected chunks
        :rtype: tuple[int, int]
        """

        acceptedTotal = 0
        rejectedTotal = 0

        for inputFileName in inputFileList:

            # read list of chunks
            chunkListFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.chunks.txt"
            result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
            if not result:
                # record error and attempt to process next file
                self.logMessage(f"makeRawVector: {fileContentOrError} - perform '--parsechunks' phase first")
            else:
                self.logMessage(f"makeRawVector: {inputFileName}")
                chunksList = json.loads(fileContentOrError)
                accepted, rejected = self.makeRawVectorPhase(chunksList, inputFileName)
                if (accepted < 0) and (rejected < 0):
                    return accepted, rejected
                acceptedTotal += accepted
                rejectedTotal += rejected

        return acceptedTotal, rejectedTotal


    def bm25ProcessPhaseAllFiles(self, inputFileList : List[str]) :
        """
        For all files create corpus of chunks.
        Each record contains <FILE_NAME>|<CHUNKID>\n<CHUNK>
        Tokenize corpus for bm25 search
        
        :param inputFileList: source file to process
        :type List[str]
        :return: 
        :rtype: 
        """

        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        corpus : List[str] = []

        for inputFileName in inputFileList:

            startFileTime = time.time()

            # read list of chunks
            chunkListFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.chunks.txt"
            result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
            if not result:
                # record error and return, bm25s index should contain all files
                self.logMessage(f"bm25Process: {fileContentOrError} - perform '--parsechunks' phase first")
                return
            else:
                self.logMessage(f"bm25Process: {inputFileName}")
                chunksList = json.loads(fileContentOrError)
                chunkId = 0
                for chunk in chunksList:
                    chunk = self.compressText(chunk, nlp)
                    outText = str(inputFileName) + '|' + str(chunkId) + "\n" + chunk
                    corpus.append(outText)
                    chunkId += 1

        # stemmer = Stemmer.Stemmer("english")
        # corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(self.bm25IndexFolder)


    def clearPhaseAllFiles(self, inputFileList : List[str]) :
        """
        For all source files remove interim files.
        
        :param inputFileList: source file to process
        :type List[str]
        :return: 
        :rtype: 
        """
        for inputFileName in inputFileList:
            rawTextFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.txt"
            OpenFile.remove(rawTextFileName)
            chunkListFileName = self.dataFolder + Path(inputFileName).name + "-data/raw.chunks.txt"
            OpenFile.remove(chunkListFileName)
            OpenFile.removedir(self.dataFolder + Path(inputFileName).name + "-data")

        result, fileNameListOrError = OpenFile.readListOfFileNames(self.bm25IndexFolder, "*.*")
        if result:
            for fileName in fileNameListOrError:
                OpenFile.remove(fileName)


    def dumpOutliersForOneQuery(self, queryService : QueryService, oneChunkQueryResultList : OneChunkQueryResultList, upperFlag : bool):

        outlierIdentifiers = queryService.getOutliersForQuery(oneQueryResultList = oneChunkQueryResultList, upper = upperFlag)
        if len(outlierIdentifiers):
            print(f"OUTLIERS=====")
            for ident in outlierIdentifiers:
                print(f"{ident}")


    def matchChunksPhase(self, queryText: str, queryService : QueryService) -> AllChunkQueryResults|None :
        """
        match chunks against known topics
        
        :param queryText: user query
        :type queryText: str
        :param queryService: Query service object
        :type queryService: QueryService
        :return: collection of all results
        :rtype: AllChunkQueryResults|None
        """

        allQueryResults = AllChunkQueryResults(
            query = [queryText],
            rrfScores = RRFScores(
                scoresDict = {}
            ),
            listQueryResults = []
        )

        if not self.initRAGcomponents():
            return allQueryResults

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return allQueryResults

        model = self.createOpenAIModel()

        # semantic search for original query
        if self.searchSemanticOriginal:
            oneQueryResultList = queryService.semanticQuery(
                query = queryText, 
                chromaCollection = chromaCollection, 
                queryLabel = "ORIG",
                maxRetrieveNumber = int(self.semanticRetrieveNumber),
                maxCutItemDistance = float(self.semanticMaxCutItemDistance)
            )
            allQueryResults.listQueryResults.append(oneQueryResultList)

#            self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for original query
        if self.searchBM25sOriginal:
            tokenList = queryService.tokenizeQuery(query = [queryText])

            oneQueryResultList, err = queryService.bm25sQuery(
                query = tokenList, 
                folderName=self.bm25IndexFolder, 
                queryLabel = "BM25SORIG", 
                bm25sRetrieveNumber = int(self.bm25sRetrieveNumber),
                bm25sMinCutOffScore = float(self.bm25sMinCutOffScore)
            )
            if not oneQueryResultList:
                # record bm25s index error and return
                self.logMessage(err)
                return None

            allQueryResults.listQueryResults.append(oneQueryResultList)

 #           self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # semantic search for multi query
        multiQueryTexts : str = ""
        if self.searchSemanticMulti:
            multiQueryTextsOrError, usage = queryService.multiQuery(queryText, model)
            if not usage:
                # record exception in LLM interface and continue
                self.logMessage(multiQueryTextsOrError)
                self.updateStats(topKey = "Multi", keyValList = [("Exception", 1)])
            else:
                multiQueryTexts = multiQueryTextsOrError
                self.addUsage(usage)
                oneQueryResultList = queryService.semanticQuery(
                    query = multiQueryTexts, 
                    chromaCollection = chromaCollection, 
                    queryLabel = "MULTI",
                    maxRetrieveNumber = int(self.semanticRetrieveNumber),
                    maxCutItemDistance = float(self.semanticMaxCutItemDistance)
                )
                allQueryResults.listQueryResults.append(oneQueryResultList)

  #          self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for multi query
        if self.searchBM25sMulti:
            if not len(multiQueryTexts):
                # Semantic Multi is False OR previous error, BM25s Multi is True - need to attempt Semantic Multi first
                multiQueryTextsOrError, usage = queryService.multiQuery(queryText, model)
                if not usage:
                    # record exception in LLM interface and continue
                    self.logMessage(multiQueryTextsOrError)
                    self.updateStats(topKey = "Multi", keyValList = [("Exception", 1)])
                else:
                    multiQueryTexts = multiQueryTextsOrError
                    self.addUsage(usage)

            if len(multiQueryTexts):
                multiTokenList = queryService.tokenizeQuery(query = multiQueryTexts)
                oneQueryResultList, err = queryService.bm25sQuery(
                    query = multiTokenList,
                    folderName = self.bm25IndexFolder, 
                    queryLabel = "BM25SMULTI", 
                    bm25sRetrieveNumber = int(self.bm25sRetrieveNumber),
                    bm25sMinCutOffScore = float(self.bm25sMinCutOffScore)
                )
                if not oneQueryResultList:
                    # record bm25s index error and return
                    self.logMessage(err)
                    return None
                allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # semantic search for rewrite query
        rewriteQueryTexts : str = ""
        if self.searchSemanticRewrite:
            rewriteQueryTextsOrError, usage = queryService.rewriteQuery(queryText, model)
            if not usage:
                # record exception in LLM interface and continue
                self.logMessage(rewriteQueryTextsOrError)
                self.updateStats(topKey = "Rewrite", keyValList = [("Exception", 1)])
            else:
                rewriteQueryTexts = rewriteQueryTextsOrError
                self.addUsage(usage)
                oneQueryResultList = queryService.semanticQuery(
                    query = rewriteQueryTexts, 
                    chromaCollection = chromaCollection, 
                    queryLabel = "REWRITE",
                    maxRetrieveNumber = int(self.semanticRetrieveNumber),
                    maxCutItemDistance = float(self.semanticMaxCutItemDistance)
                )
                allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for rewrite query
        if self.searchBM25sRewrite:
            if not len(rewriteQueryTexts):
                # Semantic Rewrite is False OR previous error, BM25s Rewrite is True - need to attempt Semantic Rewrite first
                rewriteQueryTextsOrError, usage = queryService.rewriteQuery(queryText, model)
                if not usage:
                    # record exception in LLM interface and continue
                    self.logMessage(rewriteQueryTextsOrError)
                    self.updateStats(topKey = "Rewrite", keyValList = [("Exception", 1)])
                else:
                    rewriteQueryTexts = rewriteQueryTextsOrError
                    self.addUsage(usage)

            if len(rewriteQueryTexts):
                rewriteTokenList = queryService.tokenizeQuery(query = rewriteQueryTexts)
                oneQueryResultList, err = queryService.bm25sQuery(
                    query = rewriteTokenList, 
                    folderName = self.bm25IndexFolder, 
                    queryLabel = "BM25SREWRITE", 
                    bm25sRetrieveNumber = int(self.bm25sRetrieveNumber),
                    bm25sMinCutOffScore = float(self.bm25sMinCutOffScore)
                )
                if not oneQueryResultList:
                    # record bm25s index error and return
                    self.logMessage(err)
                    return None
                allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # search for semantic HyDE query
        hydeQueryText : str = ""
        if self.searchSemanticHyDE:
            hydeQueryTextsOrError, usage = queryService.hydeQuery(queryText, model)
            if not usage:
                # record exception in LLM interface and continue
                self.logMessage(hydeQueryTextsOrError)
                self.updateStats(topKey = "HyDE", keyValList = [("Exception", 1)])
            else:
                hydeQueryText = hydeQueryTextsOrError
                self.addUsage(usage)
                oneQueryResultList = queryService.semanticQuery(
                    query = hydeQueryText, 
                    chromaCollection = chromaCollection, 
                    queryLabel = "HYDE",
                    maxRetrieveNumber = int(self.semanticRetrieveNumber),
                    maxCutItemDistance = float(self.semanticMaxCutItemDistance)
                )
                allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # search for bm25s HyDE query
        if self.searchBM25sHyDE:
            if not len(hydeQueryText):
                # Semantic HyDE is False OR previous error, BM25s HyDE is True - need to attempt Semantic HyDE first
                hydeQueryTextsOrError, usage = queryService.hydeQuery(queryText, model)
                if not usage:
                    # record exception in LLM interface and continue
                    self.logMessage(hydeQueryTextsOrError)
                    self.updateStats(topKey = "HyDE", keyValList = [("Exception", 1)])
                else:
                    self.addUsage(usage)
                    hydeQueryText = hydeQueryTextsOrError
            
            if len(hydeQueryText):
                hydeTokenList = queryService.tokenizeQuery(query = hydeQueryText)
                oneQueryResultList, err = queryService.bm25sQuery(
                    query = hydeTokenList, 
                    folderName = self.bm25IndexFolder, 
                    queryLabel = "BM25SHYDE", 
                    bm25sRetrieveNumber = int(self.bm25sRetrieveNumber),
                    bm25sMinCutOffScore = float(self.bm25sMinCutOffScore)
                )
                if not oneQueryResultList:
                    # record bm25s index error and return
                    self.logMessage(err)
                    return None
                allQueryResults.listQueryResults.append(oneQueryResultList)    

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        allQueryResults = queryService.rrfReRanking(allQueryResults)
        queryService.getOutliersFromRRF(
            allChunkQueryResults = allQueryResults, 
            iqrCoefficient = self.rrfOutlierIQRCoefficient, 
            zScoreThreshold = self.rrfOutlierZScoreThreshold
        )
        allQueryResults = self.configureOutput(allQueryResults)
        return allQueryResults


    def matchChunksPhaseAllQueries(self, queryTexts: List[str], queryService : QueryService) -> CollectionChunkQueryResults :
        """
        For all queries in the list, match chunks against known topics
        
        :param queryTexts: list of topics to match
        :type queryTexts: List[str]
        :param queryService: Query service object
        :type queryService: QueryService
        :return: collection of all query result sets
        :rtype: CollectionChunkQueryResults
        """

        collectionChunkQueryResults = CollectionChunkQueryResults(
            rrfCutOffValue = float(self.rrfCutOffValue),
            rrfOutlierZScoreThreshold = self.rrfOutlierZScoreThreshold,
            rrfOutlierIQRCoefficient = self.rrfOutlierIQRCoefficient
        )

        for oneQuery in queryTexts:

            print(f"Processing query: {oneQuery}")

            allChunkQueryResults = self.matchChunksPhase(queryText = oneQuery, queryService = queryService)
            if allChunkQueryResults:
                collectionChunkQueryResults.listAllQueryResults.append(allChunkQueryResults)

        return collectionChunkQueryResults


    def configureOutput(self, allQueryResults : AllChunkQueryResults) -> AllChunkQueryResults:
        """
        configure output as per 'outputNumber' configuration 
        Leave RRF results as is.
        
        :param allQueryResults: query results
        :type allQueryResults: AllIndexerQueryResults
        :return: query results updated with rank
        :rtype: AllQueryResults
        """

        allChunkQueryResultsNew = AllChunkQueryResults(
            query = allQueryResults.query,
            rrfScores = allQueryResults.rrfScores,
            listQueryResults = []
        )
        for queryResultList in allQueryResults.listQueryResults:
            oneChunkQueryResultList = OneChunkQueryResultList(
                label = queryResultList.label, 
                query = queryResultList.query,
                result_dict = {}
            )
            count = 0
            for key in queryResultList.result_dict.keys():
                if count >= int(self.outputNumber):
                    break
                oneChunkQueryResultList.appendQueryResult(key, queryResultList.result_dict[key])
                count += 1
            allChunkQueryResultsNew.listQueryResults.append(oneChunkQueryResultList)
        return allChunkQueryResultsNew


    def showConfiguration(self, CliCall : bool) :
        if CliCall:
            print(f"Status file:\t{self.statusFileName}")
            print(f"Documents:\t{self.documentFolder}")
            print(f"RAG database:\t{self.ragDatapath}")
            print(f"Interim data:\t{self.dataFolder}")
            print(f"BM25s folder:\t{self.bm25IndexFolder}")
            print(f"Output file:\t{self.outputFileName}")
        else:
            self.logMessage(f"Status file: {self.statusFileName}")
            self.logMessage(f"Documents: {self.documentFolder}")
            self.logMessage(f"RAG database: {self.ragDatapath}")
            self.logMessage(f"Interim data: {self.dataFolder}")
            self.logMessage(f"BM25s folder: {self.bm25IndexFolder}")
            self.logMessage(f"Output file: {self.outputFileName}")


    def logMessage(self, msg : str | List[str]):
        """
        Logs status and updates status file

        Args:
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            if type(msg) == str:
                self.statusLog.append(msg)
            else:
                for strOut in msg:
                    self.statusLog.append(strOut)
            with open(self.statusFileName, "w") as jsonOut:
                formattedOut = json.dumps(self.statusLog, indent=2)
                jsonOut.write(formattedOut)


    def threadWorker(self):
        """
        Workflow to perform query. 
        
        Args:
            None
        
        Returns:
            None

        """

        totalStart = time.time()

        self.logMessage(f"Workflow started: {self.taskId}")
        self.inWorkflow = True

        if len(self.source):
            fileList = self.source
        else:
            self.source = self.formFileList()
            fileList = self.source

        if len(fileList) == 1:
            self.logMessage(f"Discovered {len(fileList)} file for processing.")
        else:
            self.logMessage(f"Discovered {len(fileList)} files for processing.")

        #------------------loadDocument---------------------

        if self.loadDocument:
            startTime = time.time()
            self.loadDocumentPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Load", keyValList = [("Time", time.time() - startTime)])

        # ---------------parseChunks ---------------

        if self.parseChunks:
            startTime = time.time()
            self.parseChunksPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Chunking", keyValList = [("Time", time.time() - startTime)])

        # ------------makeRawVector----------------------

        if self.makeRawVector:
            startTime = time.time()
            accepted, rejected = self.makeRawVectorPhaseAllFiles(inputFileList = fileList)
            if (accepted < 0) and (rejected < 0):
                self.logMessage(f"Workflow completed with errors: {self.taskId}")        
                self.inWorkflow = False
                return
            self.updateStats(topKey = "Vectorizing", keyValList = [("Time", time.time() - startTime), ("Vectors Accepted", accepted), ("Vectors Rejected", rejected)])

        # ------------bm25Process----------------------

        if self.bm25Process:
            startTime = time.time()
            self.bm25ProcessPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "BM25 Process", keyValList = [("Time", time.time() - startTime)])

        # --------------search------------------

        if self.search:
            startTime = time.time()
            queryService = QueryService()
            collectionChunkQueryResults = self.matchChunksPhaseAllQueries(queryTexts = self.query, queryService = queryService)

            # output results files
            with open(self.outputFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
                jsonOut.writelines(collectionChunkQueryResults.model_dump_json(indent=2))

#            msgList = self.outputRRFInfo(allQueryResults.rrfScores)
#            print(msgList)
#            self.logMessage(msgList)

            self.updateStats(topKey = "Matching", keyValList = [("Time", time.time() - startTime)])

        # -------------- clear ---------------

        if self.clear:
            startTime = time.time()
            self.clearPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Clearing", keyValList = [("Time", time.time() - startTime)])


        self.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart)])

        self.logMessage(self.formatAllStats())
        self.logMessage(f"Workflow completed: {self.taskId}")
        self.inWorkflow = False
