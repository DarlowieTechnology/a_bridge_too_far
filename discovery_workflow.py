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
from resultsQueryClasses import SEARCH, OneQueryChunkResult, OneQueryResultList, IdentifierQueryResults, RRFScores, AllQueryResults
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

knownTopics = [
#    "pipe engineering",
#    "penetration test results",
    "medical notes"
]



class DiscoveryWorkflow(WorkflowBase):

    # app specific configuration
    documentFolder : str = Field(default = "documents/", description="Source document folder")
    dataFolder : str = Field(default = "discoverydata/", description="Intermediate data folder")
    bm25IndexFolder : str = Field(default = "__combined.bm25/", description="bm25 index folder")
    bm25CorpusFileName : str = Field(default = "corpus.jsonl", description="bm25 corpus file")
    fileExtensions : List[str] = Field(default = ["*.txt"], description="List of source file allowed file name extensions")
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
    makeRawVector : bool = Field(default = False, description="Vectorize chunks in intermediate table")
    bm25Process : bool = Field(default = False, description="Create bm25 index")
    matchChunks : bool = Field(default = False, description="Match chunks against known topics")
    vectorize : bool = Field(default = False, description="Vectorize in specialized tables")
    verify : bool = Field(default = False, description="Verify results against known data set")
    returnResults : bool = Field(default = False, description="Collect test results")
    clear : bool = Field(default = False, description="Clear Intermediate data")

    # retrieval configuration
    semanticRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with semantic query")
    semanticMaxCutItemDistance: float  = Field(default = 1.0, description="Maximum distance in semantic search")
    bm25sRetrieveNumber : int = Field(default = 512, description="Number of items retrieved with bm25s query")
    rrfCutOffValue : float = Field(default = 1.0, description="Reciprocal Rank Fusion value cut off")

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
        if not self.semanticRetrieveNumber in range(0, 513):
            raise ValueError(f'Number of semantic search items is invalid')
        if not (self.semanticMaxCutItemDistance >= 0 and self.semanticMaxCutItemDistance <= 1.0):
            raise ValueError(f'Maximum distance of semantic search items is invalid')
        if not self.bm25sRetrieveNumber in range(0, 513):
            raise ValueError(f'Number of bm25s search items is invalid')
        if not (self.rrfCutOffValue >= 0 and self.rrfCutOffValue <= 1.0):
            raise ValueError(f'Reciprocal Rank Fusion (RRF) cut off value is invalid')



        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        # workflow actions
        self.loadDocument = configCollection["loadDocument"]
        self.parseChunks = configCollection["parseChunks"]
        self.makeRawVector = configCollection["makeRawVector"]
        self.bm25Process = configCollection["bm25Process"]
        self.matchChunks = configCollection["matchChunks"]
        self.vectorize = configCollection["vectorize"]
        self.verify = configCollection["verify"]
        self.returnResults = configCollection["returnResults"]
        self.clear = configCollection["clear"]

        self.stripWhiteSpace = configCollection["stripWhiteSpace"]
        self.convertToLower = configCollection["convertToLower"]
        self.convertToASCII = configCollection["convertToASCII"]
        self.singleSpaces = configCollection["singleSpaces"]

        # other app-specific configuration
        self.documentFolder = configCollection["DISCOVdocumentFolder"]
        self.dataFolder = configCollection["DISCOVdataFolder"]
        self.bm25IndexFolder = configCollection["DISCOVbm25IndexFolder"]
        self.fileExtensions = configCollection["fileExtensions"]
        self.chunkSize = configCollection["chunkSize"]
        self.chunkOverlap = configCollection["chunkOverlap"]

        self.semanticRetrieveNumber = configCollection["semanticRetrieveNumber"]
        self.semanticMaxCutItemDistance = configCollection["semanticMaxCutItemDistance"]
        self.bm25sRetrieveNumber = configCollection["bm25sRetrieveNumber"]
        self.rrfCutOffValue = configCollection["rrfCutOffValue"]

        self.stats = {}

        # manually call model validator
        self.discoveryWorkflow_verify_configuration()


    def processText(self, textIn : str) -> str:
        """
        Process text per flags (strip, lower, conver to ASCII, single space)
        
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
        Load text from JSON file
        
        :param inputFile: file name
        :type inputFile: str
        :return: Text
        :rtype: str
        """

        with open(inputFile, "r" , encoding="utf-8", errors="ignore") as JsonIn:
            docs = json.load(JsonIn)

        try:
            dataframe = pd.DataFrame.from_dict(docs)
            print(dataframe)
        except ValueError as e:
            dataframe = pd.DataFrame.from_dict(docs, orient="index")

        pd.set_option('display.max_colwidth', None)
        dataframe = str(dataframe)
        dataframe = self.processText(dataframe)

        return dataframe




    def vectorize(self, allTopicMatches : AllTopicMatches)  -> tuple[int, int]:
        """
        Add all chunks to vector database.
        
        :param allTopicMatches: chunks to add
        :type allTopicMatches: AllTopicMatches
        :return: tuple of accepted and rejected chunks
        :rtype: tuple[int, int]
        """

        accepted = 0
        rejected = 0

        if not self.initRAGcomponents():
            self.fails.append(f"vectorize: RAG database failed to initialize")
            return accepted, rejected            

        for topic in allTopicMatches.topic_dict.keys():
            matchingChunks = allTopicMatches.topic_dict[topic]
            if len(matchingChunks.chunk_list):

                # ChromaDB does not accept table names with spaces
                collectionName = topic.replace(" ", "_")

                chromaCollection = self.openOrCreateCollection(collectionName = collectionName, createFlag = True)        
                if not chromaCollection:
                    return 0, 0

                ids : list[str] = []
                docs : list[str] = []
                docMetadata : list[str] = []
                embeddings = []

                for chunkInfo in matchingChunks.chunk_list:
                    chunkInfo = ChunkInfo.model_validate(chunkInfo)

                    recordHash = hash(chunkInfo.chunk)
                    uniqueId = str(chunkInfo.uuid)
                    queryResult = chromaCollection.get(ids=[uniqueId])
                    if (len(queryResult["ids"])) :

                        existingRecord = ChunkInfo.model_validate_json(queryResult["documents"][0])
                        existingHash = hash(existingRecord.chunk)

                        if recordHash == existingHash:
                            rejected += 1
        #                    msg = f"Skip {uniqueId}"
        #                    self.workerSnapshot(msg)
                            continue
                        else:
                            accepted += 1
                            msg = f"Replacing {uniqueId}"
                            self.workerSnapshot(msg)
                            chromaCollection.delete(ids=[uniqueId])
                    else:
                        accepted += 1
                        msg = f"Adding {uniqueId}"
                        self.workerSnapshot(msg)

                    vectorSource = chunkInfo.model_dump_json()

                    ids.append(uniqueId)
                    docs.append(vectorSource)
                    metadataDict = {}
                    metadataDict["recordType"] = type(chunkInfo).__name__
                    metadataDict["document"] = chunkInfo.docName
                    docMetadata.append( metadataDict )
#                    embeddings.append(self.embeddingFunction([vectorSource])[0])

                if len(ids):
                    chromaCollection.add(
#                        embeddings=embeddings,
                        documents=docs,
                        ids=ids,
                        metadatas=docMetadata
                    )

        return accepted, rejected


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


    def updateStats(self, keyValList : List[tuple[str, int]]) :
        """
        Update internal statistics. Attempt to update first, create key second.
        
        :param keyValList:  list of stats tuples (key-val)
        :type keyValList: List[tuple[str, int]]
        :return: None
        :rtype: None
        """

        for key, value in keyValList:
            try:
                prevVal = self.stats[key]
                self.stats[key] = prevVal + value
            except Exception:
                self.stats[key] = value


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
        for rank in rrfScores.scoresDict.keys():
            if rank < self.rrfCutOffValue:
                break
            identifierQueryResults = rrfScores[rank]
            ident = identifierQueryResults.identifier
            msg = f"{rank:.4f} {ident}"
            print(msg)
            outStrings.append(msg)
#            print(oneResult.chunk)

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
            self.updateStats([("Files", 1), ("Unknown MIME type", 1)])
            return 0

        # make data path if does not exist
        Path(dataFolder).mkdir(parents=True, exist_ok=True)

        if mime_type == "application/pdf":
            textCombined = self.loadPDFPyPDFLoader(inputFileName)
            if not textCombined:
                textCombined = self.loadPDFpymupdf4llm(inputFileName)
            self.updateStats([("Files", 1), ("Length", len(textCombined)), ("PDF", 1), ("PDF Length", len(textCombined))])

        if mime_type in ["application/json"]:
            textCombined = self.loadJSON(inputFileName)
            self.updateStats([("Files", 1), ("Length", len(textCombined)), ("JSON", 1), ("JSON Length", len(textCombined))])

        if mime_type in ["text/css", "text/csv", "text/html", "text/markdown", "text/plain"]:
            textCombined = self.loadText(inputFileName)
            self.updateStats([("Files", 1), ("Length", len(textCombined)), ("Other text", 1), ("Other text Length", len(textCombined))])

        fullOutputFileName = dataFolder + "/" + outputFileName
        with open(fullOutputFileName, "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)

        return len(textCombined)


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
        retTexts = []
        for text in texts:
            text = text.strip()
            text = text.lower()
            text = anyascii(text)
            text = " ".join(text.split())
            retTexts.append(text)

        self.updateStats([("Chunks", len(retTexts))])
        return retTexts


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
            recordHash = hashFunc.hexdigest() + "-" + str(chunkId)

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

        self.updateStats([("Vectors Accepted", accepted), ("Vectors Rejected", rejected)])
        return accepted, rejected


    def matchChunksPhase(self, queryTexts: List[str], queryService : QueryService) -> AllQueryResults :
        """
        match chunks against known topics
        
        :param queryTexts: list of topics to match
        :type queryTexts: List[str]
        :param queryService: Query service object
        :type queryService: QueryService
        :return: collection of all results
        :rtype: AllQueryResults
        """

        allQueryResults = AllQueryResults(
            result_lists = [],
            rrfScores = RRFScores(
                scoresDict = {}
            )
        )

        if not self.initRAGcomponents():
            return {}

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return {}

        model = self.createOpenAIModel()

        queryTexts = ["medical research into human dermoids"]

        
#        queryTexts, usage = queryService.multiQuery(queryTexts, model)
#        self.addUsage(usage)

        fullQueryList = []
        for item in queryTexts:
            fullQueryList.append(item)
            hyde_text, usage = queryService.hydeQuery(item, model)
            fullQueryList.append(hyde_text)
            self.addUsage(usage)
        queryTexts = fullQueryList

#        queryTexts, usage = QueryService.rewriteQuery(self, queryTexts, model)
#        self.addUsage(usage)

#        print(queryTexts)
#        print(self.totalUsageFormat())

        processedFullQueryList = []
        for item in fullQueryList:
            item = item.strip()
            item = item.lower()
            item = anyascii(item)
            item = " ".join(item.split())
            processedFullQueryList.append(item)
        queryTexts = processedFullQueryList

        semanticQueryResultList = queryService.semanticQuery(
            query = queryTexts, 
            chromaCollection = chromaCollection, 
            queryLabel = "semantic result",
            maxRetrieveNumber = self.semanticRetrieveNumber,
            maxCutItemDistance = self.semanticMaxCutItemDistance)
        allQueryResults.result_lists.append(semanticQueryResultList)

        tokenList = queryService.tokenizeQuery(query = queryTexts, tokenizerTypes = TOKENIZERTYPES.STOPWORDSEN | TOKENIZERTYPES.STEMMER)

        bm25sFolder = self.dataFolder + self.bm25IndexFolder
        bm25sQueryResultList = queryService.bm25sQuery(query = tokenList, folderName=bm25sFolder, queryLabel = "BM25S results", bm25sRetrieveNumber = self.bm25sRetrieveNumber)
        allQueryResults.result_lists.append(bm25sQueryResultList)

        allQueryResults = queryService.rrfReRanking(allQueryResults)

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
        self.stage = "started"

        fileList = self.formFileList()

        msg = f"Discovered {len(fileList)} files for processing."
        self.workerSnapshot(msg)

        # make root data path if does not exist
        Path(self.dataFolder).mkdir(parents=True, exist_ok=True)

        #------------------loadDocument---------------------

        if self.loadDocument:
            startTime = time.time()
            for inputFileName in fileList:
                intermediateDataFolder = self.dataFolder + "/" + str(inputFileName) + "-data"
                self.loadDocumentPhase(inputFileName, intermediateDataFolder, "raw.txt")
            endTime = time.time()
            self.updateStats([("Time Load Documents", endTime - startTime)])

        # ---------------parseChunks ---------------

        if self.parseChunks:
            startTime = time.time()
            for inputFileName in fileList:

                # read raw text into string
                rawTextFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.txt"
                result, fileContentOrError = OpenFile.open(filePath = rawTextFileName, readContent = True)
                if not result:
                    msg = f"parseChunks: {fileContentOrError} - perform 'loadDocument' action first"
                    self.workerSnapshot(msg)
                else:
                    chunksList = self.parseChunksPhase(fileContentOrError)
                    chunkListFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.chunks.txt"

                    with open(chunkListFileName, "w" , encoding="utf-8", errors="ignore") as jsonOut:
                        jsonOut.writelines(json.dumps(chunksList, indent=2))
            endTime = time.time()
            self.updateStats([("Time Chunking", endTime - startTime)])

        # ------------makeRawVector----------------------

        if self.makeRawVector:
            startTime = time.time()
            for inputFileName in fileList:

                # read list of chunks
                chunkListFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.chunks.txt"
                result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
                if not result:
                    msg = f"matchChunks: {fileContentOrError} - perform 'parseChunks' action first"
                    self.workerSnapshot(msg)
                else:
                    chunksList = json.loads(fileContentOrError)
                    accepted, rejected = self.makeRawVectorPhase(chunksList, inputFileName)
            endTime = time.time()
            self.updateStats([("Time Vectorizing", endTime - startTime)])

        # ------------bm25Process----------------------

        if self.bm25Process:
            startTime = time.time()

            # make bm25 index folder if does not exist
            Path(self.dataFolder + self.bm25IndexFolder).mkdir(parents=True, exist_ok=True)

            # Load spaCy English model
            nlp = spacy.load("en_core_web_sm")

            corpus = []
            for inputFileName in fileList:

                startFileTime = time.time()

                # read list of chunks
                chunkListFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.chunks.txt"
                result, fileContentOrError = OpenFile.open(filePath = chunkListFileName, readContent = True)
                if not result:
                    msg = f"bm25Process: {fileContentOrError} - perform 'parseChunks' action first"
                    self.workerSnapshot(msg)
                else:
                    chunksList = json.loads(fileContentOrError)
                    chunkId = 0
                    for chunk in chunksList:
                        chunk = self.compressText(chunk, nlp)
                        outText = str(inputFileName) + '--' + str(chunkId) + "\n" + chunk
                        corpus.append(outText)
                        chunkId += 1
                print(f"{inputFileName}   {time.time() - startFileTime}  added {len(chunksList)} chunks")


            # stemmer = Stemmer.Stemmer("english")
            # corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

            corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
            retriever = bm25s.BM25(corpus=corpus)
            retriever.index(corpus_tokens)
            retriever.save(self.dataFolder + self.bm25IndexFolder)

            endTime = time.time()
            self.updateStats([("Time bm25Process", endTime - startTime)])

        # --------------matchChunks------------------

        if self.matchChunks:
            startTime = time.time()
            queryService = QueryService()
            allQueryResults = self.matchChunksPhase(queryTexts = knownTopics, queryService = queryService)

            msgList = self.outputRRFInfo(allQueryResults.rrfScores)
#            self.workerSnapshot(msgList)

            endTime = time.time()
            self.updateStats([("Time Matching", endTime - startTime)])

        # -------------- clear ---------------

        if self.clear:
            startTime = time.time()

            for inputFileName in fileList:
                rawTextFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.txt"
                OpenFile.remove(rawTextFileName)
                chunkListFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.chunks.txt"
                OpenFile.remove(chunkListFileName)

            result, fileNameListOrError = OpenFile.readListOfFileNames(self.dataFolder + self.bm25IndexFolder, "*.*")
            if result:
                for fileName in fileNameListOrError:
                    OpenFile.remove(fileName)

            endTime = time.time()
            self.updateStats([("Time Clearing", endTime - startTime)])

        totalEnd = time.time()
        self.updateStats([("Time Total", totalEnd - totalStart)])

        msg = f"{pprint(self.stats)}"
        self.workerSnapshot(msg)
        return





        totalCounts = [0] * 4
        chunks = []

        for inputFileName in fileList:
            counts, allTopicMatches = self.processOneFile(inputFileName)
            totalCounts[0] += counts[0]
            totalCounts[1] += counts[1]
            totalCounts[2] += counts[2]
            totalCounts[3] += counts[3]
            for key in allTopicMatches.topic_dict.keys():
                matchingChunks = allTopicMatches.topic_dict[key]
                for chunk in matchingChunks.chunk_list:
                    chunks.append(chunk)

        score = totalCounts[0] - totalCounts[1] - totalCounts[2] * 0.5
        if score < 0:
            score = 0
        if totalCounts[3] > 0:
            scorePerCent = (score/totalCounts[3]) * 100
        else:
            scorePerCent = 0

        for chunk in chunks:
            self.workerSnapshot(str(chunk))




        # ---------------matchChunks -----------------

        if "matchChunks" in self.context and self.context["matchChunks"]:

            startTime = time.time()
            result = True
            if 'chunksList' not in locals():
                # if this is a separate step - read list of chunks
                result, fileContentOrError = OpenFile.open(filePath = self.context['chunksListRaw'], readContent = True)
                if not result:
                    msg = f"matchChunks: {fileContentOrError} - perform 'parseChunks' action first"
                    self.workerSnapshot(msg)
                else:
                    chunksList = json.loads(fileContentOrError)

            if result:

                matchResults, runSingleUsage = self.matchAllChunks(chunksList, knownTopics)

                allTopicMatches = AllTopicMatches(topic_dict = {})
                for topic in knownTopics:
                    matchingChunks = MatchingChunks(topic = topic, chunk_list =[])
                    allTopicMatches.topic_dict[topic] = matchingChunks

                for topic in matchResults.keys():
                    if topic in allTopicMatches.topic_dict.keys():
                        matchingChunks = allTopicMatches.topic_dict[topic]
                        for chunk in matchResults[topic]:
                            chunkInfo = ChunkInfo(uuid = uuid4(), docName = self.context['inputFileName'], chunk = chunk)
                            matchingChunks.chunk_list.append(chunkInfo)

                with open(self.context["matchJSON"], "w", encoding="utf-8", errors="ignore") as jsonOut:
                    jsonOut.writelines(allTopicMatches.model_dump_json(indent=2))

                self.addUsage(runSingleUsage)
                endTime = time.time()
                msg = f"matchChunks: {self.usageFormat(runSingleUsage)} Time: {(endTime - startTime):.2f} seconds"
                self.workerSnapshot(msg)


        # ------------vectorize----------------------

        if "vectorize" in self.context and self.context["vectorize"]:

            startTime = time.time()
            result = True
            if 'allTopicMatches' not in locals():
                # if this is a separate step - read match file
                result, fileContentOrError = OpenFile.open(filePath = self.context['matchJSON'], readContent = True)
                if not result:
                    msg = f"vectorize: {fileContentOrError} - perform 'matchChunks' action first"
                    self.workerSnapshot(msg)
                else:
                    allTopicMatches = AllTopicMatches.model_validate_json(fileContentOrError)

            if result:
                accepted, rejected = self.vectorize(allTopicMatches)
                endTime = time.time()
                msg = f"vectorize: accepted {accepted}  rejected {rejected}. Time: {(endTime - startTime):.2f} seconds"
                self.workerSnapshot(msg)


        # ------------verify----------------------

        if "verify" in self.context and self.context["verify"]:

            startTime = time.time()
            result = True
            totalScore = 0

            result, fileContentOrError = OpenFile.open(filePath = self.context['verifyInfo'], readContent = True)
            if not result:
                msg = f"verify: {fileContentOrError} - supply 'verify info' file"
                self.workerSnapshot(msg)
            else:
                verifyInfo = json.loads(fileContentOrError)
                if 'allTopicMatches' not in locals():
                    # if this is a separate step - read match file
                    result, fileContentOrError = OpenFile.open(filePath = self.context['matchJSON'], readContent = True)
                    if not result:
                        msg = f"verify: {fileContentOrError} - perform 'matchChunks' action first"
                        self.workerSnapshot(msg)
                    else:
                        allTopicMatches = AllTopicMatches.model_validate_json(fileContentOrError)

                if result:
                    for item in verifyInfo:
                        expectedTopic = item["name"]
                        expected = item["value"]
                        totalScore += expected
                        if expectedTopic in allTopicMatches.topic_dict.keys():
                            matchingChunks = allTopicMatches.topic_dict[expectedTopic]
                            number = len(matchingChunks.chunk_list)
                            if expected >= number:
                                counts[0] += number
                                counts[1] += (expected - number)
                            if expected < number:
                                counts[0] += expected
                                counts[2] = number - expected
                        else:
                            if expected > 0:
                                counts[1] += expected

                    scoreForFile = counts[0] - counts[1] - counts[2] * 0.5
                    counts[3] = totalScore
                    if scoreForFile < 0:
                        scoreForFile = 0
                    if totalScore > 0:
                        scorePerCent = (scoreForFile/totalScore) * 100
                    else:
                        scorePerCent = 0
                    endTime = time.time()
                    msg = f"verify:  {counts[0]}|{counts[1]}|{counts[2]}  score : {scorePerCent:.2f}%.  Time: {(endTime - startTime):.2f} seconds"
                    self.workerSnapshot(msg)


        # -------------- return results ---------------

        if "returnResults" in self.context and self.context["returnResults"]:

            startTime = time.time()
            result = True

            if 'allTopicMatches' not in locals():
                # if this is a separate step - read match file
                result, fileContentOrError = OpenFile.open(filePath = self.context['matchJSON'], readContent = True)
                if not result:
                    msg = f"returnResults: {fileContentOrError} - perform 'matchChunks' action first"
                    self.workerSnapshot(msg)
                else:
                    allTopicMatches = AllTopicMatches.model_validate_json(fileContentOrError)
                
                if result:
                    return counts, allTopicMatches


        # -------------- clear ---------------

        if "clear" in self.context and self.context["clear"]:

            startTime = time.time()
            OpenFile.remove(self.context["rawtext"])
            OpenFile.remove(self.context["chunksListRaw"])
            OpenFile.remove(self.context["matchJSON"])
            endTime = time.time()
            msg = f"clear: Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)






















        # ---------------completed ---------------

        msg = f"TotalCounts: {totalCounts}    score:{scorePerCent:.2f} %"
        self.workerSnapshot(msg)

        with open("fails.json", "w" , encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(self.getFails(), indent=2))

        totalEnd = time.time()
        self.stage = "completed"
        msg = f"Workflow completed. {self.totalUsageFormat()}. Total time {(totalEnd - totalStart):.2f} seconds."
        self.workerSnapshot(msg)
