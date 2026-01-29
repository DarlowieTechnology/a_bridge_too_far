#
# Discovery workflow class used by Django app and command line
#
from typing import List, Dict
from logging import Logger
import json
import re
import time
from pathlib import Path

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
from common import COLLECTION, RecordCollection, AllTopicMatches, SectionInfo
from workflowbase import WorkflowBase 

class DiscoveryWorkflow(WorkflowBase):

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


    def matchSectionOllama(self, doc : str, topics: List[str]) -> tuple[bool, RunUsage] :
        """
        match section against known topics
        
        :param docs: section to match
        :type doc: str
        :param topics: topics to match
        :type topics: list of str
        :return: Tuple of result and LLM usage
        :rtype: tuple[bool, RunUsage]
        """

        prompt = f"{doc}"
        ollModel = self.createOpenAIChatModel()
        runTotalUsage = RunUsage()
        retList = []

        for topic in topics:

            systemPrompt = f"""
            You are an expert in English text analysis.
            The user prompt contains English text. 
            Output True or False if topic below semantically match the text in user prompt:
            {topic}
            """
            agent = Agent(ollModel,
                        output_type=bool,
                        system_prompt = systemPrompt)
            try:
                result = agent.run_sync(prompt)
                runTotalUsage += result.usage()
                if result.output:
                    retList.append(topic)
            except pydantic_ai.exceptions.UnexpectedModelBehavior:
                msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
                self.workerSnapshot(msg)
            except ValidationError as e:
                msg = f"Exception: ValidationError {e}"
                self.workerSnapshot(msg)

        return retList, runTotalUsage


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

        for topic in allTopicMatches.topic_dict.keys():
            matchingSections = allTopicMatches.topic_dict[topic]
            if len(matchingSections.section_list):
                collectionName = topic.replace(" ", "")
                chromaCollection = self.openOrCreateCollection(collectionName = collectionName, createFlag = True)        
                if not chromaCollection:
                    return 0, 0

                ids : list[str] = []
                docs : list[str] = []
                docMetadata : list[str] = []
                embeddings = []
                accepted = 0
                rejected = 0

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



