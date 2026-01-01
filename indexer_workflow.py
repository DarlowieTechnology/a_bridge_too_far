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
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from jira import JIRA

from openai import OpenAI
from mistralai import Mistral

from langchain_community.document_loaders.pdf import PyPDFLoader

import Stemmer
import bm25s

from anyascii import anyascii



# local
from common import COLLECTION, RecordCollection
from workflowbase import WorkflowBase 

class IndexerWorkflow(WorkflowBase):

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context=context, logger=logger, createCollection=True)


    def loadPDF(self, inputFile : str) -> str :
        """
        Use PyDPF to load PDF and extract all text
        convert all characters to ASCII
        merge all pages and remove extra whitespace 
        
        Args:
            inputFile (str) - PDF file name

        Returns:
            str - combined text with pages separated by \n
        """

        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()

        textCombined = ""
        for page in docs:
            pageContent = page.page_content.strip().lower()
            pageContent = anyascii(pageContent)
            pageContent = " ".join(pageContent.split())
            textCombined += pageContent
        return textCombined




    def preprocessReportRawText(self, rawText : str) -> dict[str, str] :
        """
        Split raw text into pages using separator.
        Regexp pattern contains at least two named match groups: group IDENTIFIERGROUP is an issue identifier, group TERMINATORGROUP is the issue section terminator.
        If group IDENTIFIERGROUP match, this is a start of the issue
        If group TERMINATORGROUP match, processing is terminated.

        Args:
            rawText (str) - Text to parse
        
        Returns:
            dict of pages with unique key derived from separator
        """

        compiledPattern = re.compile(self.context["issuePattern"])
        start = -1
        end = -1
        dictIssues = {}
        uniqIdx = 0
        prevMatch = None

        for match in re.finditer(compiledPattern, rawText) :

            end = match.start()

            if prevMatch and prevMatch.group("IDENTIFIERGROUP"):
                print(f"match: {prevMatch.start()}, IDENTIFIERGROUP: {prevMatch.group('IDENTIFIERGROUP')} start: {start}   end:{end}")

            if start > 0 and end > 0:
                # insert previous page with unique key
                key = prevMatch.group("IDENTIFIERGROUP")

                if key in dictIssues:
                    if key:
                        key = key + str(uniqIdx)
                        uniqIdx += 1
                if key:
                    dictIssues[key] = rawText[start:end]
            
            if match.group("TERMINATORGROUP"):
                # current match is terminator
                break

            start = match.start()
            prevMatch = match

        # process last match only if it is not a terminator 
        if match.group("TERMINATORGROUP"):
            return dictIssues    

        end = len(rawText)
        dictIssues[prevMatch.group(0)] = rawText[start:end]
        return dictIssues


    def parseFallback(self, docs : str, ClassTemplate : BaseModel) -> BaseModel :
        """
        Fallback on regexp parsing of source data if LLM call failed to extract
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            BaseModel
        """

        msg = f"Parse fallback to regexp"
        self.workerSnapshot(msg)
        if self.context["extractPattern"] and self.context["assignList"]:
            compiledExtract = re.compile(self.context["extractPattern"])
            match = re.search(compiledExtract, docs)
            if match:
                oneIssue = ClassTemplate()
                for i in range(len(self.context["assignList"])):
                    attrName = self.context["assignList"][i]
                    setattr(oneIssue, attrName, match.group(attrName))
                return oneIssue
            else:
                msg = f"regexp parser produced no match"
                self.workerError(msg)
                return None

        msg = f"regexp parser not configured"
        self.workerError(msg)
        return None


    def parseIssueOllama(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, Usage] :
        """
        Use Ollama host and Pydantic AI Agent to extracts one ClassTemplate structured record. 
        ClassTemplate is a subclass of Pydantic BaseModel.
        
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

        ollModel = OpenAIModel(model_name=self.context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self.context["llmBaseUrl"]))

        agent = Agent(ollModel,
                    output_type=ClassTemplate,
                    system_prompt = systemPrompt)
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

        # attempt regexp match only if LLM match failed
        return self.parseFallback(docs, ClassTemplate), None


    def parseIssueGemini(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, Usage] :
        """
        Use Google Gemini AI Agent to extracts one ClassTemplate structured record. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        """

        api_key = self.config["gemini_key"]

        openAIClient = OpenAI(
            api_key=api_key,
            base_url=self.config["gemini_base_url"]
        )

        try:
            completion = openAIClient.beta.chat.completions.parse(
                model = self.context["llmGeminiVersion"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": f"JSON schema for the ClassTemplate model you must use as context for what information is expected:  {json.dumps(ClassTemplate.model_json_schema(), indent=2)}"},
                    {"role": "user", "content": f"{docs}"},
                ],
                response_format=ClassTemplate,
            )
            oneIssue = completion.choices[0].message.parsed
            if oneIssue:
                for attr in oneIssue.__dict__:
                    if oneIssue.__dict__[attr]:
                        oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                        oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
                oneIssue = ClassTemplate.model_validate_json(oneIssue.model_dump_json())

                # map Open AI usage to Pydantic usage            
                usage = Usage()
                usage.requests = 1
                usage.request_tokens = completion.usage.prompt_tokens
                usage.response_tokens = completion.usage.completion_tokens
                   
                return oneIssue, usage
            else:
                msg = f"Gemini API error"
                self.workerError(msg)
            
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = "Exception: {e}"
            self.workerSnapshot(msg)

        # attempt regexp match only if LLM match failed
        return self.parseFallback(docs, ClassTemplate), None

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

        recordCollection = RecordCollection(
            report = str(Path(self.context['inputFileName']).name),
            finding_dict = {}
        )

        for key in dictText:
            startOneIssue = time.time()

            if self.context["llmProvider"] == "Ollama":
                oneIssue, usageStats = self.parseIssueOllama(dictText[key], ClassTemplate)
            if self.context["llmProvider"] == "Gemini":
                time.sleep(self.config["gemini_time_delay"])
                oneIssue, usageStats = self.parseIssueGemini(dictText[key], ClassTemplate)
            if not oneIssue:
                continue

            recordCollection[key] = oneIssue

            endOneIssue = time.time()
            if usageStats:
                msg = f"{key}. {usageStats.requests} request(s). {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens. Time: {(endOneIssue - startOneIssue):9.4f} seconds."
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += usageStats.request_tokens
                self.context["llmresponsetokens"] += usageStats.response_tokens
            else:
                msg = f"{key}. {(endOneIssue - startOneIssue):9.4f} seconds."
            self.workerSnapshot(msg)

        return recordCollection


    
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
        jira_server = self.config["jira_url"]
        jira_user = self.config["Jira_user"]
        jira_api_token = self.config["Jira_api_token"]

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

        jql_query = f'project = {self.context["inputFileName"]}'
        recordCollection = RecordCollection(finding_dict = {})

        # Fetch issues from Jira
        # default maxResults is 50, we need more than that
        issues = jira.search_issues(jql_query, maxResults=self.config["jira_max_results"], json_result = True)
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


    def bm25sAddReportToCorpus(self, corpus : list[str], issues: RecordCollection, ClassTemplate : BaseModel) -> List[str] :
        """
        For each issue add lower case identifier and title to corpus

        Args:
            corpus - list of strings to add to
            issues (RecordCollection) - issues extracted from data source 
            ClassTemplate (BaseModel) - description of structured data
        
        Returns:
            updated corpus
        """

        for key in issues.finding_dict:
            reportItem = ClassTemplate.model_validate(issues.finding_dict[key])
            issueText = reportItem.bm25s()
            corpus.append(issueText.lower())
        return corpus


    def bm25sProcessCorpus(self, corpus : list[str], folderName: str) -> List[List[str]] :
        """
        Tokenize corpus
        Store bm25s index in a folder

        Args:
            corpus (list[str]) - list of strings representing identifier and title of all issues across documents
            folderName (str) - name of folder to save bm25s index
        Returns:
            bm25s compatible index
        """

#        stemmer = Stemmer.Stemmer("english")
 #       corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(folderName)
        return corpus_tokens


    def vectorize(self, recordCollection : RecordCollection, ClassTemplate : BaseModel) -> tuple[int, int] :
        """
        Add all structured records to vector database.
        Before vectorization improve English text
        1. Lowercase
        2. Drop stop words

        
        Args:
            recordCollection (RecordCollection) - all items to add 
            ClassTemplate (BaseModel) - issue template

        Returns:
            accepted (int) - number of records accepted to database (new or updated)
            rejected (int) - number of records rejected from database (existing)
        """

        if not self.chromaClient:
            msg = f"Cannot find Chroma DB Persistent Client"
            self.workerError(msg)
            return 0, 0

        if self.context["JiraExport"]:
            collectionName = COLLECTION.JIRA.value
        else:
            collectionName = COLLECTION.ISSUES.value

        chromaCollection = self.openOrCreateCollection(collectionName = collectionName, createFlag = True)
        if not chromaCollection:
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
            queryResult = chromaCollection.get(ids=[uniqueId])
            if (len(queryResult["ids"])) :

                existingRecordJSON = json.loads(queryResult["documents"][0])
                existingRecord = ClassTemplate.model_validate(existingRecordJSON)
                existingHash = hash(existingRecord)

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

#            if self.context["JiraExport"]:
#                vectorSource = reportItem.model_dump_json()
#            else:
#                vectorSource = reportItem.title
            vectorSource = reportItem.model_dump_json()

            ids.append(uniqueId)
            docs.append(vectorSource)
            metadataDict = {}
            metadataDict["recordType"] = type(reportItem).__name__
            metadataDict["document"] = recordCollection.report
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


    def writeFinalJSON(self, recordCollection : RecordCollection) :
        """
        Write final version of JSON as a result of LLM text comprehension.
        This JSON can be vectorized
        
        Args:
            recordCollection (RecordCollection) - all items

        Returns:
            None
        """
        with open(self.context["finalJSON"], "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines('\n"report": {recordCollection.report}\n')
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
        if self.context["JiraExport"]:

            totalStart = time.time()
            startTime = totalStart
            self.context["stage"] = "vectorizing"

            recordCollection = self.jiraExport(issueTemplate)
            accepted, rejected = self.vectorize(recordCollection, issueTemplate)
            if self.context["stage"] == "error":
                return

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
            self.workerSnapshot(msg)

            # ---------------stage completed ---------------

            self.context["stage"] = "completed"
            totalEnd = time.time()
            msg = f"Processing completed. Total time {(totalEnd - totalStart):9.4f} seconds."
            self.workerSnapshot(msg)
            return

        # ---------------stage readpdf ---------------
        totalStart = time.time()
        startTime = totalStart

        self.context["stage"] = "reading document"
        self.workerSnapshot(None)

        textCombined = self.loadPDF(self.context["inputFileName"])
        with open(self.context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)
        endTime = time.time()

        inputFileBaseName = str(Path(self.context['inputFileName']).name)
        msg = f"Read input document {inputFileBaseName}. Time: {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage preprocess raw text ---------------

        startTime = time.time()
        self.context["stage"] = "pre-processing document"
        self.workerSnapshot(None)

        dictRawIssues = self.preprocessReportRawText(textCombined)
        rawJSONFileName = self.context["rawJSON"]
        with open(rawJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
            jsonOut.writelines(json.dumps(dictRawIssues, indent=2))
        endTime = time.time()

        rawTextFromPDFBaseName = str(Path(self.context['rawtextfromPDF']).name)
        msg = f"Preprocessed raw text {rawTextFromPDFBaseName}. Found {len(dictRawIssues)} potential issues. Time: {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage fetch issues ---------------

        startTime = time.time()
        self.context["stage"] = "fetching issues"
        self.workerSnapshot(None)

        recordCollection = self.parseAllIssues(self.context["inputFileName"], dictRawIssues, issueTemplate)

        # ignore error state coming from item parser
        #if self.context["stage"] == "error":
        #    return
        self.writeFinalJSON(recordCollection)

        endTime = time.time()

        finalJSONBaseName = str(Path(self.context['finalJSON']).name)
        msg = f"Fetched {recordCollection.objectCount()} Wrote final JSON {finalJSONBaseName}. {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)

        # ---------------stage bm25s index ---------------

        startTime = time.time()
        self.context["stage"] = "bm25s index"
        self.workerSnapshot(None)

        self.bm25sProcessIssueText(recordCollection, issueTemplate)

        endTime = time.time()

        msg = f"Created BM25s index in {self.context["bm25sJSON"]}. {(endTime - startTime):9.4f} seconds"
        self.workerSnapshot(msg)


        # ---------------stage vectorize --------------

        startTime = time.time()
        self.context["stage"] = "vectorizing"
        self.workerSnapshot(None)

        accepted, rejected = self.vectorize(recordCollection, issueTemplate)
        if self.context["stage"] == "error":
            return

        endTime = time.time()
        msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
        self.workerSnapshot(msg)

        # ---------------stage completed ---------------

        self.context["stage"] = "completed"

        totalEnd = time.time()
        msg = f"Processing completed. {self.context["llmrequests"]} requests. {self.context["llmrequesttokens"]} request tokens. {self.context["llmresponsetokens"]} response tokens. Total time {(totalEnd - totalStart):9.4f} seconds."
        self.workerSnapshot(msg)
