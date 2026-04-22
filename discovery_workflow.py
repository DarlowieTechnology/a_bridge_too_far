#
# Discovery workflow class used by Django app and command line
#
from typing import List, Dict
from typing_extensions import Self
from logging import Logger
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

from openai import Model, AsyncOpenAI, ChatCompletion

import pandas as pd

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from jira import JIRA

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
from common import COLLECTION, TOKENIZERTYPES, ConfigCollection, MatchingChunks, AllTopicMatches, ChunkInfo, OpenFile
from resultsQueryClasses import SEARCH, OneQueryChunkResult, OneChunkQueryResultList, IdentifierQueryResults, RRFScores, AllChunkQueryResults
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

    # app specific configuration
    documentFolder : str = Field(default = "", description="Source document folder")
    dataFolder : str = Field(default = "", description="Intermediate data folder")
    bm25IndexFolder : str = Field(default = "", description="bm25 index folder")
    fileExtensions : List[str] = Field(default = [], description="List of source file allowed file name extensions")
    chunkSize : int = Field(default = 512, description="Chunk size for source documents")
    chunkOverlap : int = Field(default = 32, description="Chunk overlap for source documents")

    # text processing flags
    stripWhiteSpace : bool = Field(default = False, description="Strip excessive whitespace characters from source text")
    convertToLower : bool = Field(default = False, description="Covert all characters in source text to lowercase")
    convertToASCII : bool = Field(default = False, description="Covert all characters in source text to ASCII")
    singleSpaces : bool = Field(default = False, description="Replace multiple space characters with single space in source text")

    # workflow actions
    loadDocument : bool = Field(default = False, description="Load text from source documents")
    parseChunks : bool = Field(default = False, description="Create chunks out of text")
    makeRawVector : bool = Field(default = False, description="Vectorize chunks in vector table")
    bm25Process : bool = Field(default = False, description="Create bm25 index")
    matchChunks : bool = Field(default = False, description="Match chunks against known topics")
    verify : bool = Field(default = False, description="Verify results against known data set")
    returnResults : bool = Field(default = False, description="Collect test results")
    clear : bool = Field(default = False, description="Clear Intermediate data")

    # search configuration
    knownTopics : List[str] = Field(default = [], description="List of known topics to use in queries")
    searchSemanticOriginal : bool = Field(default = True, description="Perform original semantic query")
    searchBM25sOriginal : bool = Field(default = True, description="Perform original bm25s query")
    searchSemanticMulti : bool = Field(default = True, description="Perform semantic query on multi transform")
    searchBM25sMulti : bool = Field(default = True, description="Perform bm25s query on multi transform")
    searchSemanticRewrite : bool = Field(default = True, description="Perform semantic query on rewrite transform")
    searchBM25sRewrite : bool = Field(default = True, description="Perform bm25s query on rewrite transform")
    searchSemanticHyDE : bool = Field(default = True, description="Perform semantic query on HyDE transform")
    searchBM25sHyDE : bool = Field(default = True, description="Perform bm25s query on HyDE transform")

    # retrieval configuration
    semanticRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with semantic query")
    semanticMaxCutItemDistance: float  = Field(default = 1.0, description="Maximum distance in semantic search")
    bm25sRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with bm25s query")
    bm25sMinCutOffScore : float = Field(default = 0.0, description="Minimum bm25s score cut off")
    rrfCutOffValue : float = Field(default = 1.0, description="Reciprocal Rank Fusion (RRF) value cut off")
    rrfOutlierZScoreThreshold : float = Field(default = 1.5, description="Threshold for outlier z-score")
    outputNumber : int = Field(default = 1, description="Maximum number of items to return")

    outputFileName : str = Field(default = "", description="File name for results")

    stats : Dict[str, int] = Field(default = {}, description="Run statistics")

    @model_validator(mode='after')
    def discoveryWorkflow_verify_configuration(self) -> Self:

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
        if not self.chunkSize in range(128, 513):
            raise ValueError(f'Chunk size is invalid')
        if not self.chunkOverlap in range(0, 65):
            raise ValueError(f'Chunk overlap is invalid')
        
        if not self.semanticRetrieveNumber in range(0, 2049):
            raise ValueError(f'Number of semantic search items is invalid')
        if not (self.semanticMaxCutItemDistance >= 0 and self.semanticMaxCutItemDistance <= 1.0):
            raise ValueError(f'Maximum distance of semantic search items is invalid')
        if not self.bm25sRetrieveNumber in range(0, 2049):
            raise ValueError(f'Number of bm25s search items is invalid')
        if not (self.bm25sMinCutOffScore >= 0):
            raise ValueError(f'bm25s score cut off value is invalid')
        if not (self.rrfCutOffValue >= 0 and self.rrfCutOffValue <= 1.0):
            raise ValueError(f'Reciprocal Rank Fusion (RRF) cut off value is invalid')
        if not (self.rrfOutlierZScoreThreshold >= 0):
            raise ValueError(f'Z Score threshold for outliers is invalid')
        if not self.outputNumber in range(1, 100):
            raise ValueError(f'output number is invalid')

        if not Path(self.outputFileName).is_file:
            raise ValueError(f'Output file name is invalid')

        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        # workflow actions
        if configCollection.keyExists("loadDocument"): 
            self.loadDocument = configCollection["loadDocument"]
        if configCollection.keyExists("parseChunks"): 
            self.parseChunks = configCollection["parseChunks"]
        if configCollection.keyExists("makeRawVector"): 
            self.makeRawVector = configCollection["makeRawVector"]
        if configCollection.keyExists("bm25Process"): 
            self.bm25Process = configCollection["bm25Process"]
        if configCollection.keyExists("matchChunks"): 
            self.matchChunks = configCollection["matchChunks"]
        if configCollection.keyExists("verify"): 
            self.verify = configCollection["verify"]
        if configCollection.keyExists("returnResults"): 
            self.returnResults = configCollection["returnResults"]
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

        # app-specific paths configuration
        self.documentFolder = configCollection["GLOBALdataFolder"] + configCollection["DISCOVdocumentFolder"]
        self.dataFolder = configCollection["GLOBALdataFolder"] + configCollection["DISCOVdocumentFolder"] + configCollection["DISCOVdataFolder"]
        self.bm25IndexFolder = configCollection["GLOBALdataFolder"] + configCollection["DISCOVdocumentFolder"] + configCollection["DISCOVbm25IndexFolder"]
        
        self.fileExtensions = configCollection["fileExtensions"]
        self.chunkSize = configCollection["chunkSize"]
        self.chunkOverlap = configCollection["chunkOverlap"]

        # search configuration
        self.knownTopics = configCollection["knownTopics"]

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
            self.rrfOutlierZScoreThreshold   = configCollection["rrfOutlierZScoreThreshold"]
        if configCollection.keyExists("outputNumber"):
            self.outputNumber = configCollection["outputNumber"]

        self.stats = {}

        if configCollection.keyExists("outputFileName"):
            self.outputFileName = configCollection["outputFileName"]

        # manually call model validator
        self.discoveryWorkflow_verify_configuration()


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


    def loadPDFPyPDFLoader(self, inputFile : str) -> str :
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
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            self.fails.append(f"loadDocument: PyPDFLoader failed to parse {inputFile}")
            return None       

        textCombined = ""
        for page in docs:
            pageContent = page.page_content
            pageContent = self.processText(pageContent)
            textCombined += " "
            textCombined += pageContent
        return textCombined


    def loadPDFpymupdf4llm(self, inputFile : str) -> str :
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
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            self.fails.append(f"loadDocument: pymupdf4llm failed to parse {inputFile}")
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


    def outputRRFInfo(self, rrfScores : RRFScores) -> List[str]:
        """
        Output RRF scores. Stop at RRF cut off value. 

        :param rrfScores:  RRFScores object
        :type rrfScores:  RRFScores
        :return: list of output strings
        :rtype: List[str]
        """

        outStrings = []
        idx = 0
        for ident in rrfScores.scoresDict.keys():
            identifierQueryResults = rrfScores.scoresDict[ident]
            if identifierQueryResults.rrfRank < self.rrfCutOffValue:
                break
            idx += 1
            if (idx >= self.outputNumber):
                print(f"{identifierQueryResults.model_dump(mode = 'python')}")
                break
            else:
                print(f"{identifierQueryResults.model_dump(mode = 'python')},")

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

        mime_type, encoding = mimetypes.guess_type(inputFileName)
        if mime_type not in acceptedMimeTypes:
            msg = f" Error: File type not supported: {mime_type}"
            self.workerSnapshot(msg)
            self.updateStats(topKey = "Load Documents", keyValList = [("Files", 1), ("Unknown MIME type", 1)])
            return 0

        # make data path if does not exist
        Path(dataFolder).mkdir(parents=True, exist_ok=True)

        if mime_type == "application/pdf":
            textCombined = self.loadPDFPyPDFLoader(inputFileName)
            if not textCombined:
                textCombined = self.loadPDFpymupdf4llm(inputFileName)
                self.updateStats(topKey = "Load Documents", keyValList = [("pymupdf4llm", 1)])
            else:
                self.updateStats(topKey = "Load Documents", keyValList = [("PyPDFLoader", 1)])
            self.updateStats(topKey = "Load Documents", keyValList = [("Files", 1), ("Length", len(textCombined)), ("PDF", 1), ("PDF Length", len(textCombined))])

        if mime_type in ["application/json"]:
            textCombined = self.loadJSON(inputFileName)
            self.updateStats(topKey = "Load Documents", keyValList = [("Files", 1), ("Length", len(textCombined)), ("JSON", 1), ("JSON Length", len(textCombined))])

        if mime_type in ["text/css", "text/csv", "text/html", "text/markdown", "text/plain"]:
            textCombined = self.loadText(inputFileName)
            self.updateStats(topKey = "Load Documents", keyValList = [("Files", 1), ("Length", len(textCombined)), ("Other text", 1), ("Other text Length", len(textCombined))])

        fullOutputFileName = dataFolder + outputFileName
        with open(fullOutputFileName, "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)

        return len(textCombined)


    def loadDocumentPhaseAllFiles(self, inputFileList : List[str]) : 
        """
        For all files load document from one of various formats and store it as plain text

        :param inputFileList: source files to add
        :type List[str]
        :return: 
        :rtype: 
        """
        for inputFileName in inputFileList:
            fullInputFileName = self.documentFolder + inputFileName
            intermediateDataFolder = self.dataFolder + inputFileName + "-data"
            self.loadDocumentPhase(fullInputFileName, intermediateDataFolder, "raw.txt")


    def parseChunksPhase(self, docs : str) -> List[str] :
        """
        split text into chunks using text splitter object
        
        :param docs: document
        :type docs: str
        :return: chunk list
        :rtype: List[str]
        """

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunkSize, chunk_overlap=self.chunkOverlap)
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
            rawTextFileName = self.dataFolder + inputFileName + "-data/raw.txt"
            result, fileContentOrError = OpenFile.open(filePath = rawTextFileName, readContent = True)
            if not result:
                msg = f"parseChunks: {fileContentOrError} - perform 'loadDocument' phase first"
                self.workerSnapshot(msg)
                return
            else:
                chunksList = self.parseChunksPhase(fileContentOrError)
                chunkListFileName = self.dataFolder + inputFileName + "-data/raw.chunks.txt"

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
                accepted += 1

                ids.append(recordHash)
                docs.append(chunk)
                metadataDict = {}
                metadataDict["document"] = str(inputFileName)
                metadataDict["runid"] = runId
                metadataDict["chunkid"] = str(chunkId)
                docMetadata.append( metadataDict )
                embedding = self.embeddingFunction([chunk])
                embeddings.append(embedding[0])

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
            chunkListFileName = self.dataFolder + inputFileName + "-data/raw.chunks.txt"
            result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
            if not result:
                msg = f"makeRawVector: {fileContentOrError} - perform 'parseChunks' phase first"
                self.workerSnapshot(msg)
            else:
                chunksList = json.loads(fileContentOrError)
                accepted, rejected = self.makeRawVectorPhase(chunksList, inputFileName)
                acceptedTotal += accepted
                rejectedTotal += rejected

        return acceptedTotal, rejectedTotal


    def bm25ProcessPhaseAllFiles(self, inputFileList : List[str]) :
        """
        For all files create corpus of chunks.
        Each record contains <FILE_NAME>--<CHUNKID>\n<CHUNK>
        Tokenize corpus for bm25 search
        
        :param inputFileList: source file to process
        :type List[str]
        :return: 
        :rtype: 
        """

        # make bm25 index folder if does not exist
        Path(self.bm25IndexFolder).mkdir(parents=True, exist_ok=True)

        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        corpus : List[str] = []

        for inputFileName in inputFileList:

            startFileTime = time.time()

            # read list of chunks
            chunkListFileName = self.dataFolder + inputFileName + "-data/raw.chunks.txt"
            result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
            if not result:
                msg = f"bm25Process: {fileContentOrError} - perform 'parseChunks' phase first"
                self.workerSnapshot(msg)
                return
            else:
                chunksList = json.loads(fileContentOrError)
                chunkId = 0
                for chunk in chunksList:
                    chunk = self.compressText(chunk, nlp)
                    outText = str(inputFileName) + '--' + str(chunkId) + "\n" + chunk
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
            rawTextFileName = self.dataFolder + inputFileName + "-data/raw.txt"
            OpenFile.remove(rawTextFileName)
            chunkListFileName = self.dataFolder + inputFileName + "-data/raw.chunks.txt"
            OpenFile.remove(chunkListFileName)

        result, fileNameListOrError = OpenFile.readListOfFileNames(self.dataFolder + self.bm25IndexFolder, "*.*")
        if result:
            for fileName in fileNameListOrError:
                OpenFile.remove(fileName)


    def dumpOutliersForOneQuery(self, queryService : QueryService, oneChunkQueryResultList : OneChunkQueryResultList, upperFlag : bool):

        outlierIdentifiers = queryService.getOutliersForQuery(oneQueryResultList = oneChunkQueryResultList, upper = upperFlag)
        if len(outlierIdentifiers):
            print(f"OUTLIERS=====")
            for ident in outlierIdentifiers:
                print(f"{ident}")


    def matchChunksPhase(self, queryTexts: List[str], queryService : QueryService) -> AllChunkQueryResults :
        """
        match chunks against known topics
        
        :param queryTexts: list of topics to match
        :type queryTexts: List[str]
        :param queryService: Query service object
        :type queryService: QueryService
        :return: collection of all results
        :rtype: AllChunkQueryResults
        """

        allQueryResults = AllChunkQueryResults(
            query = queryTexts,
            rrfScores = RRFScores(
                scoresDict = {}
            ),
            listQueryResults = []
        )

        if not self.initRAGcomponents():
            return {}

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return {}

        model = self.createOpenAIModel()

        # semantic search for original query
        if self.searchSemanticOriginal:
            oneQueryResultList = queryService.semanticQuery(
                query = queryTexts, 
                chromaCollection = chromaCollection, 
                queryLabel = "semantic original",
                maxRetrieveNumber = self.semanticRetrieveNumber,
                maxCutItemDistance = self.semanticMaxCutItemDistance
            )
            allQueryResults.listQueryResults.append(oneQueryResultList)

#            self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for original query
        if self.searchBM25sOriginal:
            tokenList = queryService.tokenizeQuery(query = queryTexts)

            # make bm25 index folder if does not exist
            Path(self.bm25IndexFolder).mkdir(parents=True, exist_ok=True)

            oneQueryResultList = queryService.bm25sQuery(
                query = tokenList, 
                folderName=self.bm25IndexFolder, 
                queryLabel = "bm25s original", 
                bm25sRetrieveNumber = self.bm25sRetrieveNumber,
                bm25sMinCutOffScore = self.bm25sMinCutOffScore
            )
            allQueryResults.listQueryResults.append(oneQueryResultList)

 #           self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # semantic search for multi query
        multiQueryTexts = []
        if self.searchSemanticMulti:
            multiQueryTexts, usage = queryService.multiQuery(queryTexts, model)
            self.addUsage(usage)
            oneQueryResultList = queryService.semanticQuery(
                query = multiQueryTexts, 
                chromaCollection = chromaCollection, 
                queryLabel = "semantic multi",
                maxRetrieveNumber = self.semanticRetrieveNumber,
                maxCutItemDistance = self.semanticMaxCutItemDistance
            )
            allQueryResults.listQueryResults.append(oneQueryResultList)

  #          self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for multi query
        if self.searchBM25sMulti:
            if not self.searchSemanticMulti:
                multiQueryTexts, usage = queryService.multiQuery(queryTexts, model)
                self.addUsage(usage)
            multiTokenList = queryService.tokenizeQuery(query = multiQueryTexts)
            oneQueryResultList = queryService.bm25sQuery(
                query = multiTokenList, 
                folderName = self.bm25IndexFolder, 
                queryLabel = "bm25s multi", 
                bm25sRetrieveNumber = self.bm25sRetrieveNumber,
                bm25sMinCutOffScore = self.bm25sMinCutOffScore)
            allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # semantic search for rewrite query
        rewriteQueryTexts = []
        if self.searchSemanticRewrite:
            rewriteQueryTexts, usage = queryService.rewriteQuery(queryTexts, model)
            self.addUsage(usage)
            oneQueryResultList = queryService.semanticQuery(
                query = rewriteQueryTexts, 
                chromaCollection = chromaCollection, 
                queryLabel = "semantic rewrite",
                maxRetrieveNumber = self.semanticRetrieveNumber,
                maxCutItemDistance = self.semanticMaxCutItemDistance)
            allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # bm25s search for rewrite query
        if self.searchBM25sRewrite:
            if not self.searchSemanticRewrite:
                rewriteQueryTexts, usage = queryService.rewriteQuery(queryTexts, model)
                self.addUsage(usage)
            rewriteTokenList = queryService.tokenizeQuery(query = rewriteQueryTexts)
            oneQueryResultList = queryService.bm25sQuery(
                query = rewriteTokenList, 
                folderName = self.bm25IndexFolder, 
                queryLabel = "bm25s rewrite", 
                bm25sRetrieveNumber = self.bm25sRetrieveNumber,
                bm25sMinCutOffScore = self.bm25sMinCutOffScore)
            allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        # search for semantic HyDE query
        hydeQueryTexts = []
        if self.searchSemanticHyDE:
            hydeQueryTexts, usage = queryService.hydeQuery(queryTexts, model)
            self.addUsage(usage)
            oneQueryResultList = queryService.semanticQuery(
                query = hydeQueryTexts, 
                chromaCollection = chromaCollection, 
                queryLabel = "semantic hyde",
                maxRetrieveNumber = self.semanticRetrieveNumber,
                maxCutItemDistance = self.semanticMaxCutItemDistance)
            allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = False)

        # search for bm25s HyDE query
        if self.searchBM25sHyDE:
            if self.searchSemanticHyDE:
                hydeQueryTexts, usage = queryService.hydeQuery(queryTexts, model)
                self.addUsage(usage)
            hydeTokenList = queryService.tokenizeQuery(query = hydeQueryTexts)
            oneQueryResultList = queryService.bm25sQuery(
                query = hydeTokenList, 
                folderName = self.bm25IndexFolder, 
                queryLabel = "bm25s hyde", 
                bm25sRetrieveNumber = self.bm25sRetrieveNumber,
                bm25sMinCutOffScore = self.bm25sMinCutOffScore)
            allQueryResults.listQueryResults.append(oneQueryResultList)

   #         self.dumpOutliersForOneQuery(queryService, oneQueryResultList, upperFlag = True)

        allQueryResults = queryService.rrfReRanking(allQueryResults)
        queryService.getOutliersFromRRF(allQueryResults, self.rrfOutlierZScoreThreshold)
        return allQueryResults


    def threadWorker(self):
        """
        Workflow to perform query. 
        
        Args:
            None
        
        Returns:
            None

        """

        totalStart = time.time()

        fileList = self.formFileList()

        msg = f"Discovered {len(fileList)} files for processing."
        self.workerSnapshot(msg)

        # make root data path if does not exist
        Path(self.dataFolder).mkdir(parents=True, exist_ok=True)

        #------------------loadDocument---------------------

        if self.loadDocument:
            startTime = time.time()
            self.loadDocumentPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Load Documents", keyValList = [("Time", time.time() - startTime)])

        # ---------------parseChunks ---------------

        if self.parseChunks:
            startTime = time.time()
            self.parseChunksPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Chunking", keyValList = [("Time", time.time() - startTime)])

        # ------------makeRawVector----------------------

        if self.makeRawVector:
            startTime = time.time()
            accepted, rejected = self.makeRawVectorPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Vectorizing", keyValList = [("Time", time.time() - startTime), ("Chunks Accepted", accepted), ("Chunks Rejected", rejected)])

        # ------------bm25Process----------------------

        if self.bm25Process:
            startTime = time.time()
            self.bm25ProcessPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "BM25 Process", keyValList = [("Time", time.time() - startTime)])

        # --------------matchChunks------------------

        if self.matchChunks:
            startTime = time.time()
            queryService = QueryService()
            allQueryResults = self.matchChunksPhase(queryTexts = self.knownTopics, queryService = queryService)

            msgList = self.outputRRFInfo(allQueryResults.rrfScores)
#            print(msgList)
#            self.workerSnapshot(msgList)

            self.updateStats(topKey = "Matching", keyValList = [("Time", time.time() - startTime)])

        # -------------- clear ---------------

        if self.clear:
            startTime = time.time()
            self.clearPhaseAllFiles(inputFileList = fileList)
            self.updateStats(topKey = "Clearing", keyValList = [("Time", time.time() - startTime)])


        # output results files
        with open(self.outputFileName, "w") as jsonOut:
            jsonOut.writelines(allQueryResults.model_dump_json(indent=2))


        self.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart)])

        pprint(self.stats)
        return

