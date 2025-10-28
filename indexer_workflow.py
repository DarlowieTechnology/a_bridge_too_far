#
# workflow class used by Django app and command line
#
import sys
import logging
from logging import Logger
import json
import tomli
import re
import time
from pathlib import Path

from pydantic import BaseModel, Field
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from openai import OpenAI
from mistralai import Mistral

from langchain_community.document_loaders.pdf import PyPDFLoader


# local
from common import ConfigSingleton, DebugUtils, ReportIssue, AllReportIssues, OpenFile
from workflowbase import WorkflowBase 


class IndexerWorkflow(WorkflowBase):

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context, logger)


    def preprocessReportRawText(self, rawText : str, pattern: str ) -> dict[str, str] :
        """
        Split raw text into pages using separator. An example of expected format of separator is `SR-102-116`. 
        
        Args:
            rawText (str) - Text to parse
            pattern (str) - re pattern for separation of pages
        
        Returns:
            dict of pages with unique key derived from separator
        """

        compiledPattern = re.compile(pattern)
        start = -1
        end = -1
        dictIssues = {}
        uniqIdx = 0
        prevMatch = None

        for match in re.finditer(compiledPattern, rawText) :
            end = match.start()
            if start > 0 and end > 0:
                # insert previous page with unique key
                key = prevMatch.group(0)
                if key in dictIssues:
                    key = key + str(uniqIdx)
                    uniqIdx += 1
                dictIssues[key] = rawText[start:end]
                
            start = match.start()
            prevMatch = match

        # process last issue
        end = len(rawText)
        dictIssues[prevMatch.group(0)] = rawText[start:end]
        return dictIssues


    def parseIssueOllama(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, Usage] :
        """
        Use Ollama host and Pydantic AI Agent to extracts one ClassTemplate structured record. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        """

        systemPrompt = f"""
        The prompt contains an issue. Here is the JSON schema for the ReportIssue model you must use as context for what information is expected:
        {json.dumps(ReportIssue.model_json_schema(), indent=2)}
        """
        prompt = f"{docs}"

        ollModel = OpenAIModel(model_name=self._config["main_llm_name"], 
                            provider=OpenAIProvider(base_url=self._config["llm_base_url"]))

        agent = Agent(ollModel,
                    output_type=ClassTemplate,
                    system_prompt = systemPrompt,
                    retries=5,
                    output_retries=5)
        try:
            result = agent.run_sync(prompt)
            oneIssue = ClassTemplate.model_validate_json(result.output.model_dump_json())
            for attr in oneIssue.__dict__:
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
            runUsage = result.usage()
    #        DebugUtils.logPydanticObject(oneIssue, "Issue")
    #        self._logger.info(runUsage)
            return oneIssue, runUsage
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            self._logger.info(f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior")
        return None, None


    def parseIssueGemini(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, Usage] :
        """
        Use Google Gemini AI Agent to extracts one ClassTemplate structured record. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        """

        api_key = self._config["gemini_key"]

        openAIClient = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

        systemPrompt = f"""
        The prompt contains an issue. Here is the JSON schema for the ReportIssue model you must use as context for what information is expected:
        {json.dumps(ReportIssue.model_json_schema(), indent=2)}
        """
        userPrompt = f"{docs}"


        completion = openAIClient.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": userPrompt},
            ],
            response_format=ClassTemplate,
        )

        oneIssue = completion.choices[0].message.parsed
        for attr in oneIssue.__dict__:
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")

        # map Open AI usage to Pydantic usage            
        usage = Usage()
        usage.requests = 1
        usage.request_tokens = completion.usage.prompt_tokens
        usage.response_tokens = completion.usage.completion_tokens
            
        return oneIssue, usage


    def parseIssueMistral(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, Usage] :
        """
        Use Mistral AI Agent to extracts one ClassTemplate structured record. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        """

        api_key = self._config["mistral_key"]

        schema = json.dumps(ClassTemplate.model_json_schema(), indent=2)
        
        with Mistral(
            api_key=api_key
        ) as mistral:

            res = mistral.chat.parse(
                model="mistral-small-latest", 
                messages=[
                    {
                        "role": "system",
                        "content": f"The prompt contains text. Here is the JSON schema for the ReportIssue model you must use as context for what information is expected: {schema}"
                    },
                    {
                        "role": "user",
                        "content": docs
                    }
                ],
                response_format = ClassTemplate
            )
            oneIssue = ClassTemplate.model_validate_json(res.choices[0].message.content)
            for attr in oneIssue.__dict__:
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")

            # map Open AI usage to Pydantic usage            
            usage = Usage()
            usage.requests = 1
            usage.request_tokens = res.usage.prompt_tokens
            usage.response_tokens = res.usage.completion_tokens

            return oneIssue, usage

        return None, None

    def parseAllIssues(
            self,
            inputFileName : str, 
            dictText: dict[str,str], 
            ClassTemplate : BaseModel) -> AllReportIssues :
        """
        Extracts ClassTemplate instances in the dict of pages using LLM. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            inputFileName (str) - original file name
            dictText (dict[str,str]) - dict of pages
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Instance of AllReportIssues
        """

        allIssues = AllReportIssues(name = inputFileName, issue_dict = {})

        for key in dictText:
            startOneIssue = time.time()

            if self._context["llmProvider"] == "Ollama":
                oneIssue, usageStats = self.parseIssueOllama(dictText[key], ClassTemplate)    
            if self._context["llmProvider"] == "Gemini":
                time.sleep(10)
                oneIssue, usageStats = self.parseIssueGemini(dictText[key], ClassTemplate)    
            if self._context["llmProvider"] == "Mistral":
                oneIssue, usageStats = self.parseIssueMistral(dictText[key], ClassTemplate)
            allIssues.issue_dict[key] = oneIssue
            endOneIssue = time.time()
            if usageStats:
                msg = f"Fetched issue {key}. {usageStats.requests} LLM requests. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens. Time: {(endOneIssue - startOneIssue):9.4f} seconds."
                self._context["llmrequests"] += 1
                self._context["llmrequesttokens"] += usageStats.request_tokens
                self._context["llmresponsetokens"] += usageStats.response_tokens
            else:
                msg = f"Fetched issue {key}. {(endOneIssue - startOneIssue):9.4f} seconds."
            self.workerSnapshot(msg)

        return allIssues

    def loadPDF(self, inputFile : str) -> str :
        """
        Use PyDPF to load PDF and extract all text

        Args:
            inputFile (str) - PDF file name

        Returns:
            str
        """

        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()

        textCombined = ""
        for page in docs:
            textCombined += "\n" + page.page_content
        return textCombined

    def vectorize(self, allIssues : AllReportIssues) :
        """
        Add all structured records to vector database
        
        Args:
            allIssues (AllReportIssues) - list of issues

        Returns:
            None
        """
        try:
            chromaClient = chromadb.PersistentClient(
                path=self._config.getAbsPath("rag_datapath"),
                settings=Settings(anonymized_telemetry=False),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        except Exception as e:
            msg = f"Error: OpenAI API exception: {e}"
            self.workerError(msg)
            return
        
        ef = OllamaEmbeddingFunction(
            model_name=self._config["rag_embed_llm"],
            url=self._config["rag_embed_url"]    
        )

        collectionName = "reportissues"
        try:
            chromaReportIssues = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
            msg = f"Opened collections REPORTISSUES with {chromaReportIssues.count()} documents."
            self.workerSnapshot(msg)
        except chromadb.errors.NotFoundError as e:
            try:
                chromaReportIssues = chromaClient.create_collection(
                    name=collectionName,
                    embedding_function=ef,
                    metadata={ "hnsw:space": self._config["rag_hnsw_space"]  }
                )
                msg = f"Created collection REPORTISSUES"
                self.workerSnapshot(msg)
            except Exception as e:
                msg = f"Error: exception creating collection REPORTISSUES: {e}"
                self.workerError(msg)
                return

        except Exception as e:
            msg = f"Error: exception opening collection REPORTISSUES: {e}"
            self.workerError(msg)
            return

        ids : list[str] = []
        docs : list[str] = []
        docMetadata : list[str] = []
        embeddings = []

        for key in allIssues.issue_dict:
            reportIssue = allIssues.issue_dict[key]
    #        self._logger.info(f"New record: {reportIssue.model_dump_json(indent=2)}")
            recordHash = hash(reportIssue)
    #        self._logger.info(f"New hash: {recordHash}")
            uniqueId = key
            queryResult = chromaReportIssues.get(ids=[uniqueId])
            if (len(queryResult["ids"])) :

                msg = f"Record found in database {reportIssue.identifier}"
                self.workerSnapshot(msg)

                existingRecordJSON = json.loads(queryResult["documents"][0])
                existingRecord = ReportIssue.model_validate(existingRecordJSON)
    #            self._logger.info(f"Existing record: {existingRecord.model_dump_json(indent=2)}")
                existingHash = hash(existingRecord)
    #            self._logger.info(f"Existing hash: {existingHash}")

                if recordHash == existingHash:
                    msg = f"Record hash match for {reportIssue.identifier} - skipping"
                    self.workerSnapshot(msg)
                    continue
                else:
                    msg = f"Record hash different for {reportIssue.identifier}"
                    self.workerSnapshot(msg)
                    chromaReportIssues.delete(ids=[uniqueId])
                    msg = f"Deleted record {reportIssue.identifier}"
                    self.workerSnapshot(msg)

            ids.append(uniqueId)
            docs.append(reportIssue.model_dump_json())
            docMetadata.append({ "docName" : recordHash } )
            embeddings.append(ef([reportIssue.description])[0])
            msg = f"Record added in database {reportIssue.identifier}."
            self.workerSnapshot(msg)

        if len(ids):
            chromaReportIssues.add(
                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=docMetadata
            )



    def threadWorker(self):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            None

        Returns:
            None
        """

        # ---------------stage readpdf ---------------

        start = time.time()
        totalStart = start

        textCombined = self.loadPDF(self._context["inputFileName"])
        with open(self._context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)
        end = time.time()

        inputFileBaseName = str(Path(self._context['inputFileName']).name)
        msg = f"Read input document {inputFileBaseName}. Time: {(end-start):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage preprocess raw text ---------------

        start = time.time()

        allReportIssues = AllReportIssues()
        pattern = allReportIssues.pattern
        dictIssues = self.preprocessReportRawText(textCombined, pattern)
        rawJSONFileName = self._context["rawJSON"]
        with open(rawJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(dictIssues, indent=2))
        end = time.time()

        rawTextFromPDFBaseName = str(Path(self._context['rawtextfromPDF']).name)
        msg = f"Preprocessed raw text {rawTextFromPDFBaseName}. Found {len(dictIssues)} potential issues. Time: {(end-start):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage fetch issues ---------------

        start = time.time()

        allReportIssues = self.parseAllIssues(self._context["inputFileName"], dictIssues, ReportIssue)
        outputJSONFileName = self._context["finalJSON"]
        with open(outputJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(allReportIssues.model_dump_json(indent=2))

        end = time.time()
        finalJSONBaseName = str(Path(self._context['finalJSON']).name)
        msg = f"Fetched {len(allReportIssues.issue_dict)}. Wrote final JSON {finalJSONBaseName}. {(end-start):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage vectorize --------------

        start = time.time()

        self.vectorize(allReportIssues)

        end = time.time()
        msg = f"Added {len(allReportIssues.issue_dict)} to vector collections ISSUES. {(end-start):9.4f} seconds"
        self.workerSnapshot(msg)


        # ---------------stage completed ---------------

        self._context["stage"] = "completed"

        totalEnd = time.time()
        msg = f"Processing completed. Total time {(totalEnd-totalStart):9.4f} seconds. LLM request tokens: {self._context["llmrequesttokens"]}. LLM response tokens: {self._context["llmresponsetokens"]}."
        self.workerSnapshot(msg)
