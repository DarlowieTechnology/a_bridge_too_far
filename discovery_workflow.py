#
# Discovery workflow class used by Django app and command line
#
from typing import List, Dict
from logging import Logger
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4
import hashlib

from pydantic import BaseModel, ConfigDict, Field, ValidationError
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
    "pipe engineering",
    "penetration test results",
    "AWS permissions"
]



class DiscoveryWorkflow(WorkflowBase):

    context : dict[str, str] = Field(default = {}, description="Context dict") 
    fails : list[str] = Field(default = [], description="Fails list") 

    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        # workflow actions
        self.context["loadDocument"] = configCollection["loadDocument"]
        self.context["parseChunks"] = configCollection["parseChunks"]
        self.context["makeRawVector"] = configCollection["makeRawVector"]
        self.context["matchChunks"] = configCollection["matchChunks"]
        self.context["vectorize"] = configCollection["vectorize"]
        self.context["verify"] = configCollection["verify"]
        self.context["returnResults"] = configCollection["returnResults"]
        self.context["clear"] = configCollection["clear"]

        # text extraction configuration
        self.context["stripWhiteSpace"] = configCollection["stripWhiteSpace"]
        self.context["convertToLower"] = configCollection["convertToLower"]
        self.context["convertToASCII"] = configCollection["convertToASCII"]
        self.context["singleSpaces"] = configCollection["singleSpaces"]

        # other app-specific configuration
        self.context["documentFolder"] = configCollection["documentFolder"]
        self.context["fileExtensions"] = configCollection["fileExtensions"]
        self.context["chunkSize"] = configCollection["chunkSize"]
        self.context["chunkOverlap"] = configCollection["chunkOverlap"]


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
        

        if "stripWhiteSpace" in self.context and self.context["stripWhiteSpace"]:
            docs = docs.strip()
        if "convertToLower" in self.context and self.context["convertToLower"]:
            docs = docs.lower()
        if "convertToASCII" in self.context and self.context["convertToASCII"]:
            docs = anyascii(docs)
        if "singleSpaces" in self.context and self.context["singleSpaces"]:
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

        if "stripWhiteSpace" in self.context and self.context["stripWhiteSpace"]:
            docs = docs.strip()
        if "convertToLower" in self.context and self.context["convertToLower"]:
            docs = docs.lower()
        if "convertToASCII" in self.context and self.context["convertToASCII"]:
            docs = anyascii(docs)
        if "singleSpaces" in self.context and self.context["singleSpaces"]:
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

        if "stripWhiteSpace" in self.context and self.context["stripWhiteSpace"]:
            dataframe = dataframe.strip()
        if "convertToLower" in self.context and self.context["convertToLower"]:
            dataframe = dataframe.lower()
        if "convertToASCII" in self.context and self.context["convertToASCII"]:
            dataframe = anyascii(dataframe)

# leave dataframe formatting intact            
#        if "singleSpaces" in self.context and self.context["singleSpaces"]:
#            dataframe = " ".join(dataframe.split())

        return dataframe
    

    def parseChunks(self, docs : str) -> List[str] :
        """
        split text into chunks using text splitter object
        
        :param docs: document
        :type docs: str
        :return: chunk list
        :rtype: List[str]
        """

        chunkSize = self.context["chunkSize"]
        chunkOverlap = self.context["chunkOverlap"]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunkSize, chunk_overlap=chunkOverlap)
        texts = text_splitter.split_text(docs)
        retTexts = []
        for text in texts:
            text = text.strip()
            text = text.lower()
            text = anyascii(text)
            text = " ".join(text.split())
            retTexts.append(text)
        return retTexts


    def makeRawVector(self, doc : List[str]) -> tuple[int, int]:
        """
        Add all chunks to raw vector database. 
        Chunk is rejected if sha256 digest and chunk sequence in the document are the same as recorded previously
        ChromaDB default embedding model is used
        
        :param doc: chunks to add
        :type List[str]
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
#        embeddings = []

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
                metadataDict["document"] = self.context['inputFileName']
                metadataDict["runid"] = runId
                metadataDict["chunkid"] = str(chunkId)
                docMetadata.append( metadataDict )
#                embedding = self.embeddingFunction([chunk])
#                embeddings.append(embedding[0])

        if len(ids):
            chromaCollection.add(
#                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=docMetadata
            )

        return accepted, rejected


    def matchAllChunks(self, doc : List[str], topics: List[str]) -> tuple[Dict[str, List[str]], RunUsage] :
        """
        match chunks against known topics
        
        :param docs: List of chunks to match
        :type doc: List[str]
        :param topic: list of topics to match
        :type topic: List[str]
        :return: Tuple of Dict and LLM usage
        :rtype: tuple[bool, RunUsage]
        """

        if not self.initRAGcomponents():
            return {}, RunUsage()

        chromaCollection = self.getChromaCollection(COLLECTION.RAWDATA.value)
        if not chromaCollection:
            return {}, RunUsage()

        query = topics[0]

        oneQueryResultList = OneQueryResultList(
            result_dict = {},
            query = query,
            searchType = SEARCH.SEMANTIC.value,
            label = ""
        )

        cutDist = 0.5
        n_results = 3
        queryResult = chromaCollection.query(query_texts=query, n_results=n_results)

        resultIdx = -1

        for distFloat in queryResult["distances"][0]:
            resultIdx += 1
            if (distFloat > cutDist) :
                break

            print(f"------dist {distFloat}-------------------")
            print(type(queryResult["documents"][0][resultIdx]))
            print(queryResult["documents"][0][resultIdx])
            print("-------------------------")
            print(type(queryResult["metadatas"][0][resultIdx]))
            print(queryResult["metadatas"][0][resultIdx])

            oneQueryResultList.appendResult(
                identifier = "",
                title = "",
                report = queryResult["metadatas"][0][resultIdx]["document"],
                score = distFloat,
                rank = resultIdx + 1
            )

        return {}, RunUsage()





        prompt = f"{doc}"
        llmModel = self.createOpenAIModel()

#        systemPrompt = f"""You are an expert in English text analysis.\
#The user prompt contains a list of English texts.\
#Match the following list of topics against every text.\
#Output a list of key-value pairs where keys are topics and values are matching texts: {topics}"""

        systemPrompt = f"""You are an expert in English text analysis.\
Match each topic in the list against the text in user prompt: {topics}.\
Output a list topics that match the text."""

        agent = Agent(llmModel,
                    output_type=List[str],
                    system_prompt = systemPrompt,
                    retries = 5
                    )

        totalResults = {}
        for topic in topics:
            totalResults[topic] = []

        print("-------total results------------")
        print(totalResults)
        print("-------------------------")

        totalUsage = RunUsage()

        countChunks = len(doc)
        nextChunk = 0
        while nextChunk < countChunks:
            endChunk = nextChunk + 1
            if endChunk > countChunks:
                endChunk = countChunks
            nextList = doc[nextChunk:endChunk]
            try:

                print("-------user prompt------------")
                print(nextList)
                print("-------------------------")

                result = agent.run_sync(nextList, message_history=[])

                print("-------result.output-----------")
                print(result.output)
                print("-------------------------")

                for topic in totalResults.keys():
                    dictResults = result.output['output']
                    if topic in dictResults:
                        totalResults[topic].append(result.output['output'][topic])
                totalUsage += result.usage()
            except Exception as e:
                msg = f"Exception: {e}"
                self.workerSnapshot(msg)
                self.fails.append(f"matchChunks: Exception '{e}' processing {self.context['inputFileName']}")
            nextChunk = endChunk

        print("-------total results------------")
        print(totalResults)
        print("-------------------------")

        return totalResults, totalUsage

        agent = Agent(llmModel,
                    output_type=Dict[str, List[str]],
                    system_prompt = systemPrompt,
                    retries = 5
                    )
        try:
            result = agent.run_sync(prompt, message_history=[])
            return result.output, result.usage()
        
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            self.fails.append(f"matchChunks: Exception '{e}' processing {self.context['inputFileName']}")

        return {}, RunUsage()


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
            self.fails.append(f"vectorize: RAG database failed to initialize for {self.context['inputFileName']}")
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
        for globName in self.context["fileExtensions"]:
            result, fileListOrError = OpenFile.readListOfFileNames(self.context["documentFolder"], globName)
            if result:
                completeFileList = completeFileList + fileListOrError
        return completeFileList


    def getFails(self) -> List[str] :
        return self.fails


    def processOneFile(self, inputFileName : str) -> tuple[List[int], AllTopicMatches]:
        """
        Process one file in workflow
        
        :param inputFileName: name of file
        :type inputFileName: str
        :return: Tuple of score result list and matched chunks
        :rtype: tuple[List[int], AllTopicMatches]
        """
        msg = f"Input file {inputFileName}"
        self.workerSnapshot(msg)

        # count results array
        counts = [0] * 4

        self.context["inputFileName"] = str(inputFileName)
        self.context["inputFileBaseName"] = str(Path(self.context["inputFileName"]).name)
        self.context["dataFolder"] = self.context["inputFileName"] + "-data"
        self.context["rawtext"] = self.context["dataFolder"] + "/raw.txt"
        self.context["chunksListRaw"] = self.context["dataFolder"] + "/raw.chunks.txt"
        self.context["matchJSON"] = self.context["dataFolder"] + "/match.json"
        self.context["verifyInfo"] = self.context["dataFolder"] + "/verify.json"

        # make data path for the document if does not exist
        Path(self.context["dataFolder"]).mkdir(parents=True, exist_ok=True)


        # ---------------loadDocument ---------------

        if "loadDocument" in self.context and self.context["loadDocument"]:
            startTime = time.time()

            mime_type, encoding = mimetypes.guess_type(self.context['inputFileName'])
            if mime_type in acceptedMimeTypes:
                self.context['mime_type'] = mime_type
                self.context['encoding'] = encoding
            else:
                msg = f" Error: File type not supported: {mime_type}"
                self.workerSnapshot(msg)
                return

            if self.context['mime_type'] == "application/pdf":
                textCombined = self.loadPDFPyPDFLoader(self.context['inputFileName'])
                if not textCombined:
                    textCombined = self.loadPDFpymupdf4llm(self.context['inputFileName'])

            if self.context['mime_type'] in ["application/json"]:
                msg = f"loadDocument: Loading JSON data"
                self.workerSnapshot(msg)
                textCombined = self.loadJSON(self.context['inputFileName'])

            if self.context['mime_type'] in ["text/css", "text/csv", "text/html", "text/markdown", "text/plain"]:
                textCombined = self.loadText(self.context['inputFileName'])

            with open(self.context["rawtext"], "w" , encoding="utf-8", errors="ignore") as rawOut:
                rawOut.write(textCombined)

            endTime = time.time()
            msg = f"loadDocument: Read <b>{len(textCombined)} bytes</b> from file <b>{self.context['inputFileName']}</b>.  Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)


        # ---------------parseChunks ---------------

        if "parseChunks" in self.context and self.context["parseChunks"]:

            startTime = time.time()
            result = True
            if 'textCombined' not in locals():
                # if this is a separate step - read raw text into string
                result, fileContentOrError = OpenFile.open(filePath = self.context['rawtext'], readContent = True)
                if not result:
                    msg = f"parseChunks: {fileContentOrError} - perform 'loadDocument' action first"
                    self.workerSnapshot(msg)
                else:
                    textCombined = fileContentOrError

            if result:

                chunksList = self.parseChunks(textCombined)

                with open(self.context["chunksListRaw"], "w" , encoding="utf-8", errors="ignore") as jsonOut:
                    jsonOut.writelines(json.dumps(chunksList, indent=2))
                endTime = time.time()
                msg = f"parseChunks: created {len(chunksList)} chunks. Time: {(endTime - startTime):.2f} seconds"
                self.workerSnapshot(msg)


        # ------------makeRawVector----------------------

        if "makeRawVector" in self.context and self.context["makeRawVector"]:

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
                accepted, rejected = self.makeRawVector(chunksList)
                endTime = time.time()
                msg = f"makeRawVector: accepted {accepted}  rejected {rejected}  Time: {(endTime - startTime):.2f} seconds"
                self.workerSnapshot(msg)



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


        # return empty results if no "return results" action
        return counts, AllTopicMatches(topic_dict = {})


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

        # ---------------completed ---------------

        msg = f"TotalCounts: {totalCounts}    score:{scorePerCent:.2f} %"
        self.workerSnapshot(msg)

        with open("fails.json", "w" , encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(self.getFails(), indent=2))

        totalEnd = time.time()
        self.stage = "completed"
        msg = f"Workflow completed. {self.totalUsageFormat()}. Total time {(totalEnd - totalStart):.2f} seconds."
        self.workerSnapshot(msg)
