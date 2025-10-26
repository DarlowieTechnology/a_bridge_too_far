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


class IndexerWorkflow(BaseModel):

    _config : ConfigSingleton
    _logger : Logger

    def __init__(self, logger):
        """
        Args:
            logger (Logger) - can originate in CLI or Django app
        """
        self._config = ConfigSingleton()
        self._logger = logger

    def workerSnapshot(self, context : dict, msg : str):
        """
        Logs status and updates status file

        Args:
            context (dict) - process context data
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            self._logger.info(msg)
            context['status'].append(msg)
        with open(context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(context, indent=2)
            jsonOut.write(formattedOut)

    def workerError(self, context : dict, msg : str):
        """
        Logs error and sets process status to error

        Args:
            context (dict) - process context data
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            self._logger.info(msg)
            context['status'].append(msg)
        context['stage'] = 'error'
        with open(context['statusFileName'], "w") as jsonOut:
            formattedOut = json.dumps(context, indent=2)
            jsonOut.write(formattedOut)


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

        completion = openAIClient.beta.chat.completions.parse(
            model="gemini-2.0-flash",
            messages=[
                {"role": "system", "content": "Extract the record information."},
                {"role": "user", "content": docs},
            ],
            response_format=ClassTemplate,
        )

        oneIssue = completion.choices[0].message.parsed
        for attr in oneIssue.__dict__:
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
            oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
        return oneIssue, None


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
            usage = Usage(request_tokens = res.usage.prompt_tokens, response_tokens = res.usage.completion_tokens)
            return oneIssue, usage

        return None, None

    def parseAllIssues(
            self,
            inputFileName : str, 
            dictText: dict[str,str], 
            context : dict, 
            ClassTemplate : BaseModel) -> AllReportIssues :
        """
        Extracts ClassTemplate instances in the dict of pages using LLM. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            inputFileName (str) - original file name
            dictText (dict[str,str]) - dict of pages
            context (dict) - context record
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Instance of AllReportIssues
        """

        allIssues = AllReportIssues(name = inputFileName, issue_dict = {})

        for key in dictText:
            startOneIssue = time.time()

            if context["llmProvider"] == "Ollama":
                oneIssue, usageStats = self.parseIssueOllama(dictText[key], ClassTemplate)    
            if context["llmProvider"] == "Gemini":
                oneIssue, usageStats = self.parseIssueGemini(dictText[key], ClassTemplate)    
            if context["llmProvider"] == "Mistral":
                oneIssue, usageStats = self.parseIssueMistral(dictText[key], ClassTemplate)
            allIssues.issue_dict[key] = oneIssue
            endOneIssue = time.time()
            if usageStats:
                msg = f"Fetched issue {key}. {usageStats.requests} LLM requests. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens. Time: {(endOneIssue - startOneIssue):9.4f} seconds."
                context["llmrequesttokens"] += usageStats.request_tokens
                context["llmresponsetokens"] += usageStats.response_tokens
            else:
                msg = f"Fetched issue {key}. {(endOneIssue - startOneIssue):9.4f} seconds."
            self.workerSnapshot(context, msg)

        return allIssues

    def loadPDF(self, context : dict, inputFile : str) -> str :
        """
        Use PyDPF to load PDF and extract all text

        Args:
            context (dict) - context record
            inputFile (str) - PDF file name

        Returns:
            str
        """

        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()
        msg = f"{inputFile} loaded. Pages: {len(docs)}"
        self.workerSnapshot(context, msg)

        textCombined = ""
        for page in docs:
            textCombined += "\n" + page.page_content
        return textCombined

    def vectorize(self, context : dict, allIssues : AllReportIssues) :
        """
        Add all structured records to vector database
        
        Args:
            context (dict) - context record
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
            self.workerError(context, msg)
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
            self.workerSnapshot(context, msg)
        except chromadb.errors.NotFoundError as e:
            try:
                chromaReportIssues = chromaClient.create_collection(
                    name=collectionName,
                    embedding_function=ef,
                    metadata={ "hnsw:space": self._config["rag_hnsw_space"]  }
                )
                msg = f"Created collection REPORTISSUES"
                self.workerSnapshot(context, msg)
            except Exception as e:
                msg = f"Error: exception creating collection REPORTISSUES: {e}"
                self.workerError(context, msg)
                return

        except Exception as e:
            msg = f"Error: exception opening collection REPORTISSUES: {e}"
            self.workerError(context, msg)
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
                self.workerSnapshot(context, msg)

                existingRecordJSON = json.loads(queryResult["documents"][0])
                existingRecord = ReportIssue.model_validate(existingRecordJSON)
    #            self._logger.info(f"Existing record: {existingRecord.model_dump_json(indent=2)}")
                existingHash = hash(existingRecord)
    #            self._logger.info(f"Existing hash: {existingHash}")

                if recordHash == existingHash:
                    msg = f"Record hash match for {reportIssue.identifier} - skipping"
                    self.workerSnapshot(context, msg)
                    continue
                else:
                    msg = f"Record hash different for {reportIssue.identifier}"
                    self.workerSnapshot(context, msg)
                    chromaReportIssues.delete(ids=[uniqueId])
                    msg = f"Deleted record {reportIssue.identifier}"
                    self.workerSnapshot(context, msg)

            ids.append(uniqueId)
            docs.append(reportIssue.model_dump_json())
            docMetadata.append({ "docName" : recordHash } )
            embeddings.append(ef([reportIssue.description])[0])
            msg = f"Record added in database {reportIssue.identifier}."
            self.workerSnapshot(context, msg)

        if len(ids):
            chromaReportIssues.add(
                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=docMetadata
            )



    def threadWorker(self, context):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            context - initial information for index run, can originate in CLI or Django app

        Returns:
            None
        """

        # ---------------stage readpdf ---------------

        start = time.time()
        totalStart = start

        context["llmrequesttokens"] = 0
        context["llmresponsetokens"] = 0
        inputFileName = context["inputFileName"]
        context["rawtextfromPDF"] = inputFileName + ".raw.txt"
        context["rawJSON"] = inputFileName + ".raw.json"
        context["finalJSON"] = inputFileName + ".json"
        context['status'] = []
        context["stage"] = "Read PDF"

        msg = f"Starting processing"
        self.workerSnapshot(context, msg)

        textCombined = self.loadPDF(context, inputFileName)
        rawTextfileName = context["rawtextfromPDF"]
        with open(rawTextfileName, "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)
        end = time.time()
        msg = f"Read input document ({inputFileName}). Wrote raw text ({rawTextfileName}). Time: {(end-start):9.4f} seconds"
        self.workerSnapshot(context, msg)

        # ---------------stage preprocess raw text ---------------

        start = time.time()
        context["stage"] = "Preprocess raw text"

        allReportIssues = AllReportIssues()
        pattern = allReportIssues.pattern
        dictIssues = self.preprocessReportRawText(textCombined, pattern)
        rawJSONFileName = context["rawJSON"]
        with open(rawJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(dictIssues, indent=2))
        end = time.time()
        msg = f"Preprocessed raw text ({rawTextfileName}). Found {len(dictIssues)} potential issues. Time: {(end-start):9.4f} seconds"
        self.workerSnapshot(context, msg)

        # ---------------stage fetch issues ---------------

        start = time.time()
        context["stage"] = "Fetch Issues"

        allReportIssues = self.parseAllIssues(inputFileName, dictIssues, context, ReportIssue)
        outputJSONFileName = context["finalJSON"]
        with open(outputJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(allReportIssues.model_dump_json(indent=2))

        end = time.time()
        msg = f"Fetched {len(allReportIssues.issue_dict)} Wrote final JSON {outputJSONFileName}. {(end-start):9.4f} seconds"
        self.workerSnapshot(context, msg)

        # ---------------stage vectorize --------------

        start = time.time()
        context["stage"] = "vectorize"

        self.vectorize(context, allReportIssues)

        end = time.time()
        msg = f"Added {len(allReportIssues.issue_dict)} to vector collections ISSUES. {(end-start):9.4f} seconds"
        self.workerSnapshot(context, msg)


        # ---------------stage completed ---------------

        context["stage"] = "completed"

        totalEnd = time.time()
        msg = f"Processing completed. Total time {(totalEnd-totalStart):9.4f} seconds. LLM request tokens: {context["llmrequesttokens"]}. LLM response tokens: {context["llmresponsetokens"]}."
        self.workerSnapshot(context, msg)
