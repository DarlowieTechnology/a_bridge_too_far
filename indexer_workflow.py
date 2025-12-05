#
# Indexer workflow class used by Django app and command line
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
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from jira import JIRA

from openai import OpenAI
from mistralai import Mistral

from langchain_community.document_loaders.pdf import PyPDFLoader

import Stemmer
import bm25s



# local
from common import RecordCollection
from workflowbase import WorkflowBase 

class IndexerWorkflow(WorkflowBase):

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context, logger)


    def preprocessReportRawText(self, rawText : str) -> dict[str, str] :
        """
        Split raw text into pages using separator. An example of expected format of separator is `SR-102-116`. 
        Regexp pattern contains two OR match groups: group one is an issue identifier, group two is the issue section terminator.
        If group one match, this is a start of the issue
        If group two match, processing is terminated.

        Args:
            rawText (str) - Text to parse
        
        Returns:
            dict of pages with unique key derived from separator
        """

        compiledPattern = re.compile(self._context["issuePattern"])
        start = -1
        end = -1
        dictIssues = {}
        uniqIdx = 0
        prevMatch = None

        for match in re.finditer(compiledPattern, rawText) :
            end = match.start()
            if start > 0 and end > 0:
                # insert previous page with unique key
                key = prevMatch.group(1)

                if key in dictIssues:
                    if key:
                        key = key + str(uniqIdx)
                        uniqIdx += 1
                if key:
                    dictIssues[key] = rawText[start:end]
            
            if match.group(2):
                # current match is terminator
                break

            start = match.start()
            prevMatch = match

        # process last match only if it is not a terminator 
        if match.group(2):
            return dictIssues    

        end = len(rawText)
        dictIssues[prevMatch.group(0)] = rawText[start:end]
        return dictIssues


    def bm25sProcessRawText(self, pagedText : dict[str, str]) -> List[List[str]] :
        """
        Prepare text for BM25 keyword search
        Lower case
        Remove English stop words
        Apply English stemming
        Store results in a folder

        Args:
            pagedText (dict[str, str]) - Text to prepare for BM25
        
        Returns:
            bm25s compatible index
        """

        corpus = []
        stemmer = Stemmer.Stemmer("english")

        for key in pagedText:
            corpus.append(pagedText[key].lower())

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(self._context["bm25sJSON"])

        return corpus_tokens



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
        The prompt contains an issue. Here is the JSON schema for the ClassTemplate model you must use as context for what information is expected:
        {json.dumps(ClassTemplate.model_json_schema(), indent=2)}
        """
        prompt = f"{docs}"

        ollModel = OpenAIModel(model_name=self._config["main_llm_name"], 
                            provider=OpenAIProvider(base_url=self._config["llm_base_url"]))

        agent = Agent(ollModel,
                    output_type=ClassTemplate,
                    system_prompt = systemPrompt,
                    retries=3,
                    output_retries=3)
        try:
            result = agent.run_sync(prompt)

            oneIssue = ClassTemplate.model_validate_json(result.output.model_dump_json())
            for attr in oneIssue.__dict__:
                if oneIssue.__dict__[attr]:
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
            runUsage = result.usage()


            return oneIssue, runUsage
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerError(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerError(msg)
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
            base_url=self._config["gemini_base_url"]
        )

        try:
            completion = openAIClient.beta.chat.completions.parse(
                model = self._context["llmGeminiVersion"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": f"JSON schema for the ClassTemplate model you must use as context for what information is expected:  {json.dumps(ClassTemplate.model_json_schema(), indent=2)}"},
                    {"role": "user", "content": f"{docs}"},
                ],
                response_format=ClassTemplate,
            )
            oneIssue = completion.choices[0].message.parsed
            if not oneIssue:
                msg = f"Gemini API error"
                self.workerError(msg)
                return None, None
            
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            return None, None

        for attr in oneIssue.__dict__:
            if oneIssue.__dict__[attr]:
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
        
        try:
            oneIssue = ClassTemplate.model_validate_json(oneIssue.model_dump_json())
        except ValidationError as e:
            msg = "Exception: {e}"
            self.workerSnapshot(msg)
            return None, None

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
                        "content": f"The prompt contains text. Here is the JSON schema for the ClassTemplate model you must use as context for what information is expected: {schema}"
                    },
                    {
                        "role": "user",
                        "content": docs
                    }
                ],
                response_format = ClassTemplate
            )
            try:
                oneIssue = ClassTemplate.model_validate_json(res.choices[0].message.content)
            except Exception as e:
                msg = f"Mistral API error: {e}"
                self.workerError(msg)
                return None, None

            for attr in oneIssue.__dict__:
                if oneIssue.__dict__[attr]:
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")

            # map Open AI usage to Pydantic usage            
            usage = Usage()
            usage.requests = 1
            usage.request_tokens = res.usage.prompt_tokens
            usage.response_tokens = res.usage.completion_tokens

            return oneIssue, usage

        msg = f"Mistral API error"
        self.workerError(msg)
        return None, None

    def parseAllIssues(self, inputFileName : str, dictText: dict[str,str], ClassTemplate : BaseModel) -> RecordCollection :
        """
        Extracts ClassTemplate instances in the dict of pages using LLM. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            inputFileName (str) - original file name
            dictText (dict[str,str]) - dict of pages
            ClassTemplate (BaseModel) - issue template

        Returns:
            RecordCollection with keys from input dict
        """

        recordCollection = RecordCollection(finding_dict = {})

        for key in dictText:
            startOneIssue = time.time()

            if self._context["llmProvider"] == "Ollama":
                oneIssue, usageStats = self.parseIssueOllama(dictText[key], ClassTemplate)
            if self._context["llmProvider"] == "Gemini":
                time.sleep(self._config["gemini_time_delay"])
                oneIssue, usageStats = self.parseIssueGemini(dictText[key], ClassTemplate)
            if self._context["llmProvider"] == "Mistral":
                oneIssue, usageStats = self.parseIssueMistral(dictText[key], ClassTemplate)
            if not oneIssue:
                continue

            recordCollection[key] = oneIssue

            endOneIssue = time.time()
            if usageStats:
                msg = f"Fetched issue {key}. {usageStats.requests} request(s). {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens. Time: {(endOneIssue - startOneIssue):9.4f} seconds."
                self._context["llmrequests"] += 1
                self._context["llmrequesttokens"] += usageStats.request_tokens
                self._context["llmresponsetokens"] += usageStats.response_tokens
            else:
                msg = f"Fetched issue {key}. {(endOneIssue - startOneIssue):9.4f} seconds."
            self.workerSnapshot(msg)

        return recordCollection


    def loadPDF(self, inputFile : str) -> str :
        """
        Use PyDPF to load PDF and extract all text

        Args:
            inputFile (str) - PDF file name

        Returns:
            str - combined text with pages separated by \n
        """

        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()

        textCombined = ""
        for page in docs:
            textCombined += "\n" + page.page_content
        return textCombined


    def jiraExport(self, ClassTemplate : BaseModel) -> RecordCollection :
        """
        Export issues from Jira project
        Transform to smaller records
        Write as final JSON for vectorization

        Args:
            ClassTemplate (BaseModel) - issue template

        Returns:
            RecordCollection - all items
        """
        jira_server = self._config["jira_url"]
        jira_user = self._config["Jira_user"]
        jira_api_token = self._config["Jira_api_token"]

        # Connect to Jira
        try:
            jira = JIRA(server=jira_server, basic_auth=(jira_user, jira_api_token))
        except Exception as e:
            msg = f"Jira API exception: {e}"
            self.workerError(msg)
            return 0
        if not jira:
            msg = f"Jira REST API connection error"
            self.workerError(msg)
            return 0

        jql_query = f'project = {self._context["inputFileName"]}'
        recordCollection = RecordCollection(finding_dict = {})

        # Fetch issues from Jira
        # default maxResults is 50, we need more than that
        issues = jira.search_issues(jql_query, maxResults=self._config["jira_max_results"], json_result = True)
        for val in issues["issues"]:
            issueTemplate = ClassTemplate(
                identifier = val["key"],
                project_key = val["fields"]["project"]["key"],
                project_name = val["fields"]["project"]["name"],
                status_category_key = val["fields"]["statusCategory"]["key"],
                priority_name = val["fields"]["priority"]["name"],
                issue_updated = val["fields"]["updated"],
                status_name = val["fields"]["status"]["name"],
                summary = val["fields"]["summary"],
                progress = val["fields"]["progress"]["progress"],
                worklog = val["fields"]["worklog"]["worklogs"]
            )
            recordCollection.finding_dict[val["key"]] = issueTemplate

        self.writeFinalJSON(recordCollection)

        return recordCollection


    def vectorize(self, recordCollection : RecordCollection, ClassTemplate : BaseModel) -> tuple[int, int] :
        """
        Add all structured records to vector database.
        Before vectorization improve English text
        1. Lowercase
        2. Drop stop words

        
        Args:
            recordCollection (RecordCollection) - all items
            ClassTemplate (BaseModel) - issue template

        Returns:
            accepted (int) - number of records accepted to database
            rejected (int) - number of records rejected from database
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
            return 0, 0
        
        ef = OllamaEmbeddingFunction(
            model_name=self._config["rag_embed_llm"],
            url=self._config["rag_embed_url"]
        )

        if self._context["JiraExport"]:
            collectionName = self._config["rag_indexer_jira"]
        else:
            collectionName = self._config["rag_indexer_reports"]
        try:
            chromaReportIssues = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
            msg = f"Opened vector collections with {chromaReportIssues.count()} documents."
            self.workerSnapshot(msg)
        except chromadb.errors.NotFoundError as e:
            try:
                chromaReportIssues = chromaClient.create_collection(
                    name=collectionName,
                    embedding_function=ef,
                    metadata={ "hnsw:space": self._config["rag_hnsw_space"]  }
                )
                msg = f"Created vector collection"
                self.workerSnapshot(msg)
            except Exception as e:
                msg = f"Error: exception creating vector collection: {e}"
                self.workerError(msg)
                return 0, 0

        except Exception as e:
            msg = f"Error: exception opening vector collection: {e}"
            self.workerError(msg)
            return 0, 0

        ids : list[str] = []
        docs : list[str] = []
        docMetadata : list[str] = []
        embeddings = []
        accepted = 0
        rejected = 0

        for key in recordCollection.finding_dict:
            reportItem = ClassTemplate.model_validate(recordCollection[key])

            recordHash = hash(reportItem)
            uniqueId = key
            queryResult = chromaReportIssues.get(ids=[uniqueId])
            if (len(queryResult["ids"])) :

                existingRecordJSON = json.loads(queryResult["documents"][0])
                existingRecord = ClassTemplate.model_validate(existingRecordJSON)
                existingHash = hash(existingRecord)

                if recordHash == existingHash:
                    rejected += 1
#                    msg = f"Existing vector hash matches for {reportItem.identifier} - skipping"
#                    self.workerSnapshot(msg)
                    continue
                else:
                    accepted += 1
                    msg = f"Existing vector hash is different for {reportItem.identifier} - replacing"
                    self.workerSnapshot(msg)
                    chromaReportIssues.delete(ids=[uniqueId])
            else:
                accepted += 1
                msg = f"No vector found for {reportItem.identifier} - adding"
                self.workerSnapshot(msg)

            vectorSource = reportItem.model_dump_json()
            ids.append(uniqueId)
            docs.append(vectorSource)
            docMetadata.append({ "recordType" : type(reportItem).__name__ } )
            embeddings.append(ef([vectorSource])[0])

        if len(ids):
            chromaReportIssues.add(
                embeddings=embeddings,
                documents=docs,
                ids=ids,
                metadatas=docMetadata
            )

        return accepted, rejected


    def writeFinalJSON(self, recordCollection : RecordCollection) :
        """
        Write final version of JSON as a result of LLM text comprehension.
        This JSON can be vectorized
        
        Args:
            recordCollection (RecordCollection) - all items
            ClassTemplate (BaseModel) - issue template

        Returns:
            None
        """
        with open(self._context["finalJSON"], "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines('{\n"finding_dict": {\n')
            idx = 0
            for key in recordCollection.finding_dict:
                jsonOut.writelines(f'"{key}" : ')
                jsonOut.writelines(recordCollection.finding_dict[key].model_dump_json(indent=2))
                idx += 1
                if (idx < len(recordCollection.finding_dict)):
                    jsonOut.writelines(',\n')
            jsonOut.writelines('}\n}\n')


    def threadWorker(self, issueTemplate : BaseModel):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            issueTemplate (BaseModel) - issue template
        
        Returns:
            None
        """

        # ---------------stage Jira export
        if self._context["JiraExport"]:

            totalStart = time.time()
            startTime = totalStart
            self._context["stage"] = "vectorizing"

            recordCollection = self.jiraExport(issueTemplate)
            accepted, rejected = self.vectorize(recordCollection, issueTemplate)
            if self._context["stage"] == "error":
                return

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
            self.workerSnapshot(msg)

            # ---------------stage completed ---------------

            self._context["stage"] = "completed"
            totalEnd = time.time()
            msg = f"Processing completed. Total time {(totalEnd - totalStart):9.4f} seconds."
            self.workerSnapshot(msg)
            return

        # ---------------stage readpdf ---------------
        totalStart = time.time()
        startTime = totalStart

        self._context["stage"] = "reading document"
        self.workerSnapshot(None)

        textCombined = self.loadPDF(self._context["inputFileName"])
        with open(self._context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)
        endTime = time.time()

        inputFileBaseName = str(Path(self._context['inputFileName']).name)
        msg = f"Read input document {inputFileBaseName}. Time: {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage preprocess raw text ---------------

        startTime = time.time()
        self._context["stage"] = "pre-processing document"
        self.workerSnapshot(None)

        dictRawIssues = self.preprocessReportRawText(textCombined)
        rawJSONFileName = self._context["rawJSON"]
        with open(rawJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(dictRawIssues, indent=2))
        endTime = time.time()

        rawTextFromPDFBaseName = str(Path(self._context['rawtextfromPDF']).name)
        msg = f"Preprocessed raw text {rawTextFromPDFBaseName}. Found {len(dictRawIssues)} potential issues. Time: {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage fetch issues ---------------

        startTime = time.time()
        self._context["stage"] = "fetching issues"
        self.workerSnapshot(None)

        recordCollection = self.parseAllIssues(self._context["inputFileName"], dictRawIssues, issueTemplate)

        # ignore error state coming from item parser
        #if self._context["stage"] == "error":
        #    return
        self.writeFinalJSON(recordCollection)

        endTime = time.time()

        finalJSONBaseName = str(Path(self._context['finalJSON']).name)
        msg = f"Fetched {recordCollection.objectCount()} Wrote final JSON {finalJSONBaseName}. {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage vectorize --------------

        startTime = time.time()
        self._context["stage"] = "vectorizing"
        self.workerSnapshot(None)

        accepted, rejected = self.vectorize(recordCollection, issueTemplate)
        if self._context["stage"] == "error":
            return

        endTime = time.time()
        msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
        self.workerSnapshot(msg)

        # ---------------stage completed ---------------

        self._context["stage"] = "completed"

        totalEnd = time.time()
        msg = f"Processing completed. {self._context["llmrequests"]} requests. {self._context["llmrequesttokens"]} request tokens. {self._context["llmresponsetokens"]} response tokens. Total time {(totalEnd - totalStart):9.4f} seconds."
        self.workerSnapshot(msg)
