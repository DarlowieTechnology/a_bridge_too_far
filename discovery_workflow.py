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

# import Stemmer
# import bm25s

from anyascii import anyascii

# local
from common import COLLECTION, ConfigCollection, MatchingChunks, AllTopicMatches, ChunkInfo, OpenFile
from resultsQueryClasses import SEARCH, OneQueryAppResult, OneQueryResultList, AllQueryResults
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
    matchChunks : bool = Field(default = False, description="Match chunks against known topics")
    vectorize : bool = Field(default = False, description="Vectorize in specialized tables")
    verify : bool = Field(default = False, description="Verify results against known data set")
    returnResults : bool = Field(default = False, description="Collect test results")
    clear : bool = Field(default = False, description="Clear Intermediate data")

    stats : Dict[str, int] = Field(default = {}, description="Run statistics")


    @model_validator(mode='after')
    def discoveryWorkflow_verify_configuration(self) -> Self:

        # verify access to Source document folder
        if not Path(self.documentFolder).is_dir:
            raise ValueError(f'Source document folder is invalid')
        # verify access to Intermediate data folder
        if not Path(self.dataFolder).is_dir:
            raise ValueError(f'Intermediate data folder is invalid')
        # verify formatting of file extension list
        for fileExt in self.fileExtensions:
            extPattern = r"\*\.[A-Za-z0-9]*$"
            if not re.match(extPattern, fileExt):
                raise ValueError(f'File extension pattern is invalid')
        if not self.chunkSize in range(128, 513):
            raise ValueError(f'Chunk size is invalid')
        if not self.chunkOverlap in range(0, 65):
            raise ValueError(f'Chunk overlap is invalid')
        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        # workflow actions
        self.loadDocument = configCollection["loadDocument"]
        self.parseChunks = configCollection["parseChunks"]
        self.makeRawVector = configCollection["makeRawVector"]
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
        self.fileExtensions = configCollection["fileExtensions"]
        self.chunkSize = configCollection["chunkSize"]
        self.chunkOverlap = configCollection["chunkOverlap"]

        self.stats = {}

        # manually call model validator
        self.discoveryWorkflow_verify_configuration()



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
            if self.stripWhiteSpace:
                pageContent = pageContent.strip()
            if self.convertToLower:
                pageContent = pageContent.lower()
            if self.convertToASCII:
                pageContent = anyascii(pageContent)
            if self.singleSpaces:
                pageContent = " ".join(pageContent.split())
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
        

        if self.stripWhiteSpace:
            docs = docs.strip()
        if self.convertToLower:
            docs = docs.lower()
        if self.convertToASCII:
            docs = anyascii(docs)
        if self.singleSpaces:
            docs = " ".join(docs.split())
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

        if self.stripWhiteSpace:
            docs = docs.strip()
        if self.convertToLower:
            docs = docs.lower()
        if self.convertToASCII:
            docs = anyascii(docs)
        if self.singleSpaces:
            docs = " ".join(docs.split())
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

        if self.stripWhiteSpace:
            dataframe = dataframe.strip()
        if self.convertToLower:
            dataframe = dataframe.lower()
        if self.convertToASCII:
            dataframe = anyascii(dataframe)

# leave dataframe formatting intact            
#        if self.singleSpaces:
#            dataframe = " ".join(dataframe.split())

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

        for key, value in keyValList:
            try:
                prevVal = self.stats[key]
                self.stats[key] = prevVal + value
            except Exception:
                self.stats[key] = value


    def loadDocumentPhase(self, inputFileName : str, dataFolder : str, outputFileName : str) -> int: 
        """
        Load document and store it in plain text format
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


    def matchChunksPhase(self, topics: List[str]) -> Dict[str, List[str]] :
        """
        match chunks against known topics
        
        :param topic: list of topics to match
        :type topic: List[str]
        :return: Dict with topics as keys, and lists of chunks as values
        :rtype: Dict[str, List[str]]
        """

        if not self.initRAGcomponents():
            return {}

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return {}

        query = topics[0]

        oneQueryResultList = OneQueryResultList(
            result_dict = {},
            query = query,
            searchType = SEARCH.SEMANTIC.value,
            label = ""
        )

        n_results = 50
#        queryResult = chromaCollection.query(query_texts=query, n_results=n_results)
        queryResult = chromaCollection.query(query_texts=["medical", "clinical", "diagnosis", "human organ", "medical procedure"], n_results=n_results)

        resultIdx = -1

        for doc in queryResult["documents"][0]:
            resultIdx += 1
            print(queryResult["metadatas"][0][resultIdx]['document'])
            print(f"\t{queryResult["documents"][0][resultIdx]}")

        return {}


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
            for inputFileName in fileList:
                startTime = time.time()
                intermediateDataFolder = self.dataFolder + "/" + str(inputFileName) + "-data"
                readLength = self.loadDocumentPhase(inputFileName, intermediateDataFolder, "raw.txt")
                endTime = time.time()
                self.updateStats([("Time Load Documents", endTime - startTime)])
 #               msg = f"loadDocument: Read <b>{readLength} bytes</b> from file <b>{inputFileName}</b>.  Time: {(endTime - startTime):.2f} seconds"
 #               self.workerSnapshot(msg)

        # ---------------parseChunks ---------------

        if self.parseChunks:
            for inputFileName in fileList:
                startTime = time.time()

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
#                    msg = f"parseChunks: created {len(chunksList)} chunks. Time: {(endTime - startTime):.2f} seconds"
#                    self.workerSnapshot(msg)

        # ------------makeRawVector----------------------

        if self.makeRawVector:
            for inputFileName in fileList:
                startTime = time.time()

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
#                    msg = f"makeRawVector: accepted {accepted}  rejected {rejected}  Time: {(endTime - startTime):.2f} seconds"
#                    self.workerSnapshot(msg)


        # --------------matchChunks------------------

        if self.matchChunks:
            startTime = time.time()
            self.matchChunksPhase(knownTopics)
            endTime = time.time()
            self.updateStats([("Time Matching", endTime - startTime)])

        # -------------- clear ---------------

        if self.clear:
            for inputFileName in fileList:
                startTime = time.time()
                rawTextFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.txt"
                OpenFile.remove(rawTextFileName)
                chunkListFileName = self.dataFolder + "/" + str(inputFileName) + "-data/raw.chunks.txt"
                OpenFile.remove(chunkListFileName)
                endTime = time.time()
                self.updateStats([("Time Clearing", endTime - startTime)])
#                msg = f"clear: Time: {(endTime - startTime):.2f} seconds"
#                self.workerSnapshot(msg)

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
