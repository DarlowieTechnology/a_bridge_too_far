#
# Discovery workflow class used by Django app and command line
#
from typing import List, Dict
from logging import Logger
import json
import re
import time
from pathlib import Path
import mimetypes
from  uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from jira import JIRA

from langchain_community.document_loaders.pdf import PyPDFLoader

import Stemmer
import bm25s

from anyascii import anyascii



# local
from common import COLLECTION, RecordCollection, MatchingSections, AllTopicMatches, SectionInfo, OpenFile
from workflowbase import WorkflowBase 

class DiscoveryWorkflow(WorkflowBase):

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
        "medical research notes",
        "pipe engineering",
        "penetration test results"
    ]

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context=context, logger=logger, createCollection=True)



    def parseSectionsOllama(self, docs : str) -> tuple[str, RunUsage] :
        """
        split text into section according to formatting
        
        :param docs: text to split into sections
        :type docs: str
        :return: Tuple of section list and LLM usage
        :rtype: tuple[str, RunUsage]
        """

        systemPrompt = f"""
        You are an expert in English text analysis.
        The prompt contains English text. 
        Split text into sections. If table of content exists use it as a guide.
        If table of content does not exist, split the text in semantically complete chunks.
        Output all the text in the section.
        Do not escape whitespace characters.
        Format output as Python list.
        If you cannot split the text, output Python list with one entry.
        """

        prompt = f"{docs}"

        ollModel = self.createOpenAIChatModel()

        agent = Agent(ollModel,
                      output_type=List,
                      system_prompt = systemPrompt)
        try:
            result = agent.run_sync(prompt)
            runUsage = result.usage()
            return result.output, runUsage
       
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)
        return None, None


    def matchSectionOllama(self, doc : str, topic: str) -> tuple[bool, RunUsage] :
        """
        match section against known topics
        
        :param docs: section to match
        :type doc: str
        :param topic: topic to match
        :type topic: str
        :return: Tuple of result and LLM usage
        :rtype: tuple[bool, RunUsage]
        """

        prompt = f"{doc}"
        ollModel = self.createOpenAIChatModel()

        systemPrompt = f"""
        You are an expert in English text analysis.
        The user prompt contains English text. 
        Output True or False if text semantically matches topic below:
        {topic}
        """

        agent = Agent(ollModel,
                    output_type=bool,
                    system_prompt = systemPrompt)
        try:
            result = agent.run_sync(prompt)
            return result.output, result.usage()
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)

        return False, RunUsage()


    def parseAttemptRecordListOllama(self, docs : str) -> tuple[List, RunUsage] :
        """
        Use Ollama host and Pydantic AI Agent to attempt extraction of repeated templates
        
        Args:
            docs (str) - text with unstructured data

        Returns:
            Tuple of record list and LLM usage
        """

        systemPrompt = f"""
        You are an expert in English text analysis.
        The prompt contains English text. 
        You must discover repeated pattern of records in the text.
        Records should start with unique identifier.
        Output semantically complete chunks related to unique identifiers you found in the text.
        Format output as Python list. 
        """
        
        prompt = f"{docs}"

        ollModel = self.createOpenAIChatModel()

        agent = Agent(ollModel,
                      output_type=List,
                      system_prompt = systemPrompt)
        try:
            result = agent.run_sync(prompt)
            runUsage = result.usage()
            return result.output, runUsage
       
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)
        return None, None


    def discoverRecordSchemaOllama(self, docs : List) -> tuple[dict, RunUsage] :

        systemPrompt = f"""
        You are an expert in English text analysis.
        The prompt contains a list of repeated records. 
        You must discover JSON schema of a single record. JSON schema must include all the fields that you find in any record.
        Format output as JSON schema. 
        Output only valid JSON, no Markdown or extra text.
        Name identifier field 'identifier'
        """
        
        prompt = f"{docs}"

        ollModel = self.createOpenAIChatModel()

        agent = Agent(ollModel,
                      system_prompt = systemPrompt)
        try:
            result = agent.run_sync(prompt)
            runUsage = result.usage()
            return result.output, runUsage
        
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)

        # attempt regexp match only if LLM match failed
        return None, None



    def parseOneRecordOllama(self, docString : str, Model: BaseModel) -> tuple[BaseModel, RunUsage] :
        """
        Use Ollama host and Pydantic AI Agent 
        
        Args:
            docs (str) - text with unstructured data
            Model (BaseModel) - optional Pydantic model

        Returns:
            LLM RunUsage
        """

        systemPrompt = f"""
        The prompt contains an issue. Here is the JSON schema for the ClassTemplate model you must use as context for what information is expected:
        {json.dumps(Model.model_json_schema(), indent=2)}
        Name identifier field key 'identifier'
        Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python.
        Do not format output.
        """
        
        prompt = f"{docString}"

        ollModel = self.createOpenAIChatModel()

        if Model:
            agent = Agent(ollModel,
                        output_type=Model,
                        system_prompt = systemPrompt)
        else:
            agent = Agent(ollModel,
                        system_prompt = systemPrompt)
        try:            
            result = agent.run_sync(prompt)
            if Model:
                oneIssue = Model.model_validate_json(result.output.model_dump_json())
            else:
                oneIssue = result.output
            runUsage = result.usage()
            return oneIssue, runUsage
        
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)
        return None, None
    

    def vectorize(self, allTopicMatches : AllTopicMatches)  -> tuple[int, int]:
        """
        Add all sections to vector database.
        
        :param allTopicMatches: sections to add
        :type allTopicMatches: AllTopicMatches
        :return: tuple of accepted and rejected sections
        :rtype: tuple[int, int]
        """


        accepted = 0
        rejected = 0

        for topic in allTopicMatches.topic_dict.keys():
            matchingSections = allTopicMatches.topic_dict[topic]
            if len(matchingSections.section_list):

                # ChromaDB does not accept table names with spaces
                collectionName = topic.replace(" ", "_")

                chromaCollection = self.openOrCreateCollection(collectionName = collectionName, createFlag = True)        
                if not chromaCollection:
                    return 0, 0

                ids : list[str] = []
                docs : list[str] = []
                docMetadata : list[str] = []
                embeddings = []

                for sectionInfo in matchingSections.section_list:
                    sectionInfo = SectionInfo.model_validate(sectionInfo)

                    recordHash = hash(sectionInfo.section)
                    uniqueId = str(sectionInfo.uuid)
                    queryResult = chromaCollection.get(ids=[uniqueId])
                    if (len(queryResult["ids"])) :

                        existingRecordJSON = json.loads(queryResult["documents"][0])
                        existingRecord = SectionInfo.model_validate(existingRecordJSON)
                        existingHash = hash(existingRecord.section)

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

                    vectorSource = sectionInfo.model_dump_json()

                    ids.append(uniqueId)
                    docs.append(vectorSource)
                    metadataDict = {}
                    metadataDict["recordType"] = type(sectionInfo).__name__
                    metadataDict["document"] = sectionInfo.docName
                    docMetadata.append( metadataDict )
                    embeddings.append(self.embeddingFunction([vectorSource])[0])

                if len(ids):
                    chromaCollection.add(
                        embeddings=embeddings,
                        documents=docs,
                        ids=ids,
                        metadatas=docMetadata
                    )

        return accepted, rejected


    def formFileList(self) -> List[str]:

        completeFileList = []
        for globName in self.context["fileExtensions"]:
            result, fileListOrError = OpenFile.readListOfFileNames(self.context["documentFolder"], globName)
            if result:
                print(type(fileListOrError))
                completeFileList = completeFileList + fileListOrError
        return completeFileList


    def processOneFile(self, inputFileName : str) -> tuple[List[int], AllTopicMatches]:
        msg = f"Input file {inputFileName}"
        self.workerSnapshot(msg)

        # count results array
        counts = [0] * 4

        self.context["inputFileName"] = str(inputFileName)
        self.context["inputFileBaseName"] = str(Path(self.context["inputFileName"]).name)
        self.context["dataFolder"] = self.context["inputFileName"] + "-data"
        self.context["rawtext"] = self.context["dataFolder"] + "/raw.txt"
        self.context["sectionListRaw"] = self.context["dataFolder"] + "/raw.sections.txt"
        self.context["matchJSON"] = self.context["dataFolder"] + "/match.json"
        self.context["verifyInfo"] = self.context["dataFolder"] + "/verify.json"

        # make data path for the document if does not exist
        Path(self.context["dataFolder"]).mkdir(parents=True, exist_ok=True)

        # ---------------loadDocument ---------------

        if "loadDocument" in self.context and self.context["loadDocument"]:
            startTime = time.time()

            mime_type, encoding = mimetypes.guess_type(self.context['inputFileName'])
            if mime_type in self.acceptedMimeTypes:
                self.context['mime_type'] = mime_type
                self.context['encoding'] = encoding
            else:
                msg = f" File type not supported: {mime_type}"
                self.workerSnapshot(msg)
                return

            if self.context['mime_type'] == "application/pdf":
                textCombined = self.loadPDF(self.context['inputFileName'])
            if self.context['mime_type'] in ["text/css", "text/csv", "text/html", "application/json", "text/markdown", "text/plain"]:
                with open(self.context["inputFileName"], "r" , encoding="utf-8", errors="ignore") as txtIn:
                    textCombined = txtIn.read()

            with open(self.context["rawtext"], "w" , encoding="utf-8", errors="ignore") as rawOut:
                rawOut.write(textCombined)

            endTime = time.time()
            msg = f"loadDocument: Read <b>{len(textCombined)} bytes</b> from file <b>{self.context['inputFileName']}</b>.  Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------parseSections ---------------

        if "parseSections" in self.context and self.context["parseSections"]:
            startTime = time.time()
            if 'textCombined' not in locals():
                # if this is a separate step - read text into string
                with open(self.context['rawtext'], "r", encoding='utf8', errors='ignore') as txtIn:
                    textCombined = txtIn.read()

            sectionListRaw, runUsageParse = self.parseSectionsOllama(textCombined)
            self.addUsage(runUsageParse)

            sectionList = []
            for pageContent in sectionListRaw:
                pageContent = pageContent.strip()
                pageContent = pageContent.lower()
                pageContent = " ".join(pageContent.split())
                pageContent = anyascii(pageContent)
                sectionList.append(pageContent)

            with open(self.context["sectionListRaw"], "w" , encoding="utf-8", errors="ignore") as summaryJSONOut:
                summaryJSONOut.writelines(json.dumps(sectionList, indent=2))
            endTime = time.time()
            msg = f"parseSections: usage: {self.usageFormat(runUsageParse)} Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)


        # ---------------matchSections -----------------

        if "matchSections" in self.context and self.context["matchSections"]:
            startTime = time.time()
            if 'sectionList' not in locals():
                # if this is a separate step - read text into string
                with open(self.context['sectionListRaw'], "r", encoding='utf8', errors='ignore') as JsonIn:
                    sectionList = json.load(JsonIn)

            allTopicMatches = AllTopicMatches(topic_dict = {})
            for topic in self.knownTopics:
                matchingSections = MatchingSections(topic = topic, section_list = [])
                allTopicMatches.topic_dict[topic] = matchingSections

            runTotalUsage = RunUsage()
            for section in sectionList:
                for topic in self.knownTopics:
                    matchResult, runSingleUsage = self.matchSectionOllama(section, topic)
                    runTotalUsage += runSingleUsage
                    if matchResult:
                        sectionInfo = SectionInfo(uuid = uuid4(), docName = self.context['inputFileName'], section = section)
                        matchingSections = allTopicMatches.topic_dict[topic]
                        matchingSections.section_list.append(sectionInfo)
                    time.sleep(5)

            with open(self.context["matchJSON"], "w", encoding="utf-8", errors="ignore") as jsonOut:
                jsonOut.writelines(allTopicMatches.model_dump_json(indent=2))

            self.addUsage(runTotalUsage)
            endTime = time.time()
            msg = f"matchSections: usage: {self.usageFormat(runTotalUsage)} Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)


        # ------------vectorize----------------------

        if "vectorize" in self.context and self.context["vectorize"]:

            startTime = time.time()
            if 'allTopicMatches' not in locals():
                # if this is a separate step - read text into string
                with open(self.context['matchJSON'], "r", encoding='utf8', errors='ignore') as JsonIn:
                    allTopicMatchesStr = json.load(JsonIn)
                    allTopicMatches = AllTopicMatches.model_validate(allTopicMatchesStr)
            accepted, rejected = self.vectorize(allTopicMatches)
            endTime = time.time()
            msg = f"vectorize: accepted {accepted}  rejected {rejected}. Time: {(endTime - startTime):.2f} seconds"
            self.workerSnapshot(msg)


        # ------------verify----------------------

        if "verify" in self.context and self.context["verify"]:

            startTime = time.time()
            totalScore = 0

            with open(self.context['verifyInfo'], "r", encoding='utf8', errors='ignore') as JsonIn:
                verifyInfo = json.load(JsonIn)
            if 'allTopicMatches' not in locals():
                # if this is a separate step - read text into string
                with open(self.context['matchJSON'], "r", encoding='utf8', errors='ignore') as JsonIn:
                    allTopicMatchesStr = json.load(JsonIn)
                    allTopicMatches = AllTopicMatches.model_validate(allTopicMatchesStr)

            for item in verifyInfo:
                expectedTopic = item["name"]
                expected = item["value"]
                totalScore += expected
                if expectedTopic in allTopicMatches.topic_dict.keys():
                    matchingSections = allTopicMatches.topic_dict[expectedTopic]
                    number = len(matchingSections.section_list)
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

        # -------------- return data ---------------

        # return all matches
        if 'allTopicMatches' not in locals():
            fileExists, strContent = OpenFile.open(filePath = self.context['matchJSON'], readContent = False)
            if fileExists:
                # return matches if they were previously created
                with open(self.context['matchJSON'], "r", encoding='utf8', errors='ignore') as JsonIn:
                    allTopicMatchesStr = json.load(JsonIn)
                    allTopicMatches = AllTopicMatches.model_validate(allTopicMatchesStr)
            else:
                allTopicMatches = AllTopicMatches(topic_dict = {})

        return counts, allTopicMatches

