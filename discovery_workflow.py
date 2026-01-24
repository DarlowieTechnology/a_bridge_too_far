#
# Discovery workflow class used by Django app and command line
#
from typing import List
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
from common import COLLECTION, RecordCollection
from workflowbase import WorkflowBase 

class DiscoveryWorkflow(WorkflowBase):

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context=context, logger=logger, createCollection=True)


    def parseAttemptRecordListOllama(self, docs : str) -> tuple[List, RunUsage] :
        """
        Use Ollama host and Pydantic AI Agent to attempt extraction of repeated templates
        
        Args:
            docs (str) - text with unstructured data

        Returns:
            LLM RunUsage
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

        # attempt regexp match only if LLM match failed
        return List(), RunUsage()


    def parseOneRecordOllama(self, docString : str) -> tuple[dict, RunUsage] :
        """
        Use Ollama host and Pydantic AI Agent 
        
        Args:
            docs (str) - text with unstructured data

        Returns:
            LLM RunUsage
        """

        systemPrompt = f"""
        You are an expert in English text analysis.
        The prompt contains English text that starts with identifier and contains fields including risk rating, status, description, recommendation.
        Output JSON record where keys are names of fields.
        Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python.
        Do not format output.
        """
        
        prompt = f"{docString}"

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
        return {}, RunUsage()
