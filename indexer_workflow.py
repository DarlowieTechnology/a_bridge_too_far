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
            str - combined text 
        """

        loader = PyPDFLoader(file_path = inputFile, mode = "page" )
        docs = loader.load()

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


    def parseIssueOllama(self, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, RunUsage] :
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

        ollModel =OpenAIChatModel(model_name=self.context["llmVersion"],
                                  provider=OllamaProvider(base_url=self.context["llmBaseUrl"]))

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
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
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
            if not oneIssue:
                continue

            recordCollection[key] = oneIssue

            endOneIssue = time.time()
            if usageStats:
                requestLabel = 'requests' if usageStats.requests > 1 else 'request'
                msg = f"Record: <b>{key}</b>. Usage: {usageStats.requests} {requestLabel}, {usageStats.input_tokens} input tokens, {usageStats.output_tokens} output tokens. Time: {(endOneIssue - startOneIssue):9.2f} seconds."
                self.context["llmrequests"] += usageStats.requests
                self.context["llminputtokens"] += usageStats.input_tokens
                self.context["llmoutputtokens"] += usageStats.output_tokens
            else:
                msg = f"Record: <b>{key}</b>. {(endOneIssue - startOneIssue):9.2f} seconds."
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
        Add each issue to corpus

        Args:
            corpus - list of strings to add to
            issues (RecordCollection) - issues extracted from data source 
            ClassTemplate (BaseModel) - description of structured data
        
        Returns:
            updated corpus
        """

        for key in issues.finding_dict:
            issue = issues.finding_dict[key]
            reportItem = ClassTemplate.model_validate(issue)
            issueText = reportItem.bm25s()
            corpus.append(issueText)

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


    def threadWorker(self, issueTemplate : BaseModel, corpus : list[str]):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            issueTemplate (BaseModel) - issue template
            global corpus for BM25
        
        Returns:
            None
        """

        totalStart = time.time()

        msg = f"Document: <b>{self.context['inputFileBaseName']}</b>"
        self.workerSnapshot(msg)
        msg = f"Raw text file: <b>{self.context["rawtextfromPDF"]}</b>"
        self.workerSnapshot(msg)
        msg = f"Raw JSON file: <b>{self.context["rawJSON"]}</b>"
        self.workerSnapshot(msg)
        msg = f"Final JSON file: <b>{self.context["finalJSON"]}</b>"
        self.workerSnapshot(msg)

        # ---------------stage Jira export
        if "JiraExport" in self.context and self.context["JiraExport"]:

            startTime = totalStart
            self.context["stage"] = "vectorizing"

            recordCollection = self.jiraExport(issueTemplate)
            accepted, rejected = self.vectorize(recordCollection, issueTemplate)
            if self.context["stage"] == "error":
                return

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
            self.workerSnapshot(msg)

            self.context["stage"] = "completed"
            totalEnd = time.time()
            msg = f"Processing completed. Total time {(totalEnd - totalStart):9.2f} seconds."
            self.workerSnapshot(msg)
            return

        # ---------------stage read pdf ---------------
        if "loadDocument" in self.context and self.context["loadDocument"]:
            startTime = totalStart

            self.context["stage"] = "reading document"
            self.workerSnapshot(None)

            textCombined = self.loadPDF(self.context["inputFileName"])
            with open(self.context["rawtextfromPDF"], "w" , encoding="utf-8", errors="ignore") as rawOut:
                rawOut.write(textCombined)
            endTime = time.time()

            inputFileBaseName = str(Path(self.context['inputFileName']).name)
            msg = f"Read input document {self.context['inputFileBaseName']}. Time: {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage preprocess raw text ---------------
        if "rawTextFromDocument" in self.context and self.context["rawTextFromDocument"]:
            startTime = time.time()
            self.context["stage"] = "pre-processing document"
            self.workerSnapshot(None)

            if 'textCombined' not in locals():
                # if this is a separate step - read extracted text file
                with open(self.context['rawtextfromPDF'], "r", encoding='utf8', errors='ignore') as txtIn:
                    textCombined = txtIn.read()
                msg = f"Read raw text from file {self.context['inputFileBaseName']}."
                self.workerSnapshot(msg)

            dictRawIssues = self.preprocessReportRawText(textCombined)
            rawJSONFileName = self.context["rawJSON"]
            with open(rawJSONFileName, "w", encoding="utf-8", errors="ignore") as jsonOut:
                jsonOut.writelines(json.dumps(dictRawIssues, indent=2))
            endTime = time.time()

            rawTextFromPDFBaseName = str(Path(self.context['rawtextfromPDF']).name)
            msg = f"Preprocessed raw text {rawTextFromPDFBaseName}. Found {len(dictRawIssues)} potential issues. Time: {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage create final JSON ---------------
        if "finalJSONfromRaw" in self.context and self.context["finalJSONfromRaw"]:
            startTime = time.time()
            self.context["stage"] = "create final JSON"
            self.workerSnapshot(None)

            if 'dictRawIssues' not in locals():
                # if this is a separate step - read raw JSON into record collection
                with open(self.context['rawJSON'], "r", encoding='utf8', errors='ignore') as jsonIn:
                    dictRawIssues = json.load(jsonIn)
                msg = f"Read {len(dictRawIssues)} raw records from file {self.context['inputFileBaseName']}."
                self.workerSnapshot(msg)

            recordCollection = self.parseAllIssues(self.context["inputFileName"], dictRawIssues, issueTemplate)

            # ignore error state coming from item parser
            #if self.context["stage"] == "error":
            #    return
            self.writeFinalJSON(recordCollection)

            endTime = time.time()

            finalJSONBaseName = str(Path(self.context['finalJSON']).name)
            msg = f"Found {recordCollection.objectCount()} records. Wrote final JSON: <b>{finalJSONBaseName}</b>. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage bm25s preparation ---------------
        if "prepareBM25corpus" in self.context and self.context["prepareBM25corpus"]:

            startTime = time.time()
            self.context["stage"] = "bm25s preparation"
            self.workerSnapshot(None)

            if 'recordCollection' not in locals():
                # if this is a separate step - read final JSON into record collection
                with open(self.context["finalJSON"], "r", encoding='utf8', errors='ignore') as jsonIn:
                    jsonStr = json.load(jsonIn)
                recordCollection = RecordCollection.model_validate(jsonStr)
                msg = f"Read {recordCollection.objectCount()} records from file {self.context['inputFileBaseName']}."
                self.workerSnapshot(msg)

            corpus = self.bm25sAddReportToCorpus(corpus, recordCollection, issueTemplate)

            endTime = time.time()

            msg = f"Added {recordCollection.objectCount()} records to BM25 corpus. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage bm25s completion ---------------
        if "completeBM25database" in self.context and self.context["completeBM25database"]:

            startTime = time.time()
            self.context["stage"] = "bm25s completion"
            self.workerSnapshot(None)

            folderName = self.context["bm25IndexFolder"]
            self.bm25sProcessCorpus(corpus, folderName)

            endTime = time.time()

            msg = f"Created BM25 database in {folderName}. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)


        # ---------------stage vectorize --------------
        if "vectorizeFinalJSON" in self.context and self.context["vectorizeFinalJSON"]:

            startTime = time.time()
            self.context["stage"] = "vectorizing"
            self.workerSnapshot(None)

            if 'recordCollection' not in locals():
                # if this is a separate step - read final JSON into record collection
                with open(self.context["finalJSON"], "r", encoding='utf8', errors='ignore') as jsonIn:
                    jsonStr = json.load(jsonIn)
                recordCollection = RecordCollection.model_validate(jsonStr)
                msg = f"Read {recordCollection.objectCount()} records from file {self.context['inputFileBaseName']}."
                self.workerSnapshot(msg)

            accepted, rejected = self.vectorize(recordCollection, issueTemplate)
            if self.context["stage"] == "error":
                return

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted} rejected {rejected}."
            self.workerSnapshot(msg)

        # ---------------stage completed ---------------

        self.context["stage"] = "completed"

        totalEnd = time.time()
        requestLabel = 'requests' if self.context["llmrequests"] > 1 else 'request'        
        msg = f"Processing completed. Usage: {self.context['llmrequests']} {requestLabel}, {self.context['llminputtokens']} input tokens, {self.context['llmoutputtokens']} output tokens. Total time {(totalEnd - totalStart):9.2f} seconds."
        self.workerSnapshot(msg)
