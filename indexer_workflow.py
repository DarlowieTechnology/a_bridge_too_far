#
# Indexer workflow class used by Django app and command line
#
from typing import List, Dict
from typing_extensions import Self
from logging import Logger
import json
import re
import time
from pathlib import Path
import hashlib

from pydantic import BaseModel, ValidationError, Field, model_validator
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

import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

import Stemmer
import bm25s

from anyascii import anyascii



# local
from common import COLLECTION, RecordCollection, ConfigCollection, DebugUtils
from workflowbase import WorkflowBase 

class IndexerWorkflow(WorkflowBase):

    session_key : str = Field(default = "", description="logger session name for Indexer CLI")

    # file names : Original -> Raw Text -> Raw JSON -> Final JSON
    #
    inputFileName : str = Field(default = "", description="Original document name")
    rawTextFromDoc : str = Field(default = "", description="Raw text name")
    rawJSON : str = Field(default = "", description="Raw JSON name")
    finalJSON : str = Field(default = "", description="Final JSON name")

    inputFileBaseName : str = Field(default = "", description="Base name of original document")
    issueTemplate : str = Field(default = "", description="Issue template name")
    bm25IndexFolder : str = Field(default = "", description="bm25s index folder")
    bm25CorpusFileName : str = Field(default = "", description="bm25 corpus file")

    # Jira configuration
    INDEXEjira_url : str = Field(default = "", description="Jira API URL")
    INDEXEjira_max_results : int = Field(default = 999, description="maximum number of Jira items to retrieve")
    INDEXEjira_export : bool = Field(default = False, description="perform Jira export")
    Jira_user : str = Field(default = "", description="Jira API user name")

    # text processing flags
    stripWhiteSpace : bool = Field(default = False, description="Strip excessive whitespace characters from source text")
    convertToLower : bool = Field(default = False, description="Covert all characters in source text to lowercase")
    convertToASCII : bool = Field(default = False, description="Covert all characters in source text to ASCII")
    singleSpaces : bool = Field(default = False, description="Replace multiple space characters with single space in source text")

    # workflow actions
    loadDocument : bool = Field(default = False, description="Load text from source documents")
    rawTextFromDocument : bool = Field(default = False, description="preprocess text from source documents")
    finalJSONfromRaw : bool = Field(default = False, description="create final JSON")
    prepareBM25corpus : bool = Field(default = False, description="prepare BM25s corpus")
    completeBM25database : bool = Field(default = False, description="complete BM25 database")
    vectorizeFinalJSON : bool = Field(default = False, description="vectorize final JSON")

    # raw text parsing support
    issuePattern : str = Field(default = "", description="issue regexp pattern")
    issueTemplate : str = Field(default = "", description="issue template name")
    extractPattern : str = Field(default = "", description="issue field extract regexp pattern")
    assignList : List[str] = Field(default = [], description="list of issue fields to assign")


    @model_validator(mode='after')
    def indexerWorkflow_verify_configuration(self) -> Self:

        # verify access to original document
        if not Path(self.inputFileName).is_file:
            raise ValueError(f'Original document file is invalid')
        # verify raw text file name
        if not Path(self.rawTextFromDoc).is_file:
            raise ValueError(f'Raw text file name is invalid')
        # verify raw JSON file name
        if not Path(self.rawJSON).is_file:
            raise ValueError(f'Raw JSON file name is invalid')
        # verify final JSON file name
        if not Path(self.finalJSON).is_file:
            raise ValueError(f'Final JSON file name is invalid')
        # verify input file base name
        if not Path(self.inputFileBaseName).is_file:
            raise ValueError(f'Base name of original document is invalid')
        # verify access to bm25s folder
        if not Path(self.bm25IndexFolder).is_dir:
            raise ValueError(f'BM25s folder is invalid')
        if not self.INDEXEjira_max_results in range(0, 1000):
            raise ValueError(f'Jira maximum results is invalid')

        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        if configCollection.keyExists("session_key"):
            self.session_key = configCollection["session_key"]

        if configCollection.keyExists("inputFileName"):
            self.inputFileName = configCollection["inputFileName"]
        if configCollection.keyExists("rawTextFromDoc"):
            self.rawTextFromDoc = configCollection["rawTextFromDoc"]
        if configCollection.keyExists("rawJSON"):
            self.rawJSON = configCollection["rawJSON"]
        if configCollection.keyExists("finalJSON"):
            self.finalJSON = configCollection["finalJSON"]

        if configCollection.keyExists("inputFileBaseName"):
            self.inputFileBaseName = configCollection["inputFileBaseName"]

        self.bm25IndexFolder = configCollection["bm25IndexFolder"]
        self.bm25CorpusFileName = configCollection["bm25CorpusFileName"]

        if configCollection.keyExists("INDEXEjira_url"):
            self.INDEXEjira_url = configCollection["INDEXEjira_url"]
        if configCollection.keyExists("INDEXEjira_max_results"):
            self.INDEXEjira_max_results = configCollection["INDEXEjira_max_results"]
        if configCollection.keyExists("INDEXEjira_export"):
            self.INDEXEjira_export = configCollection["INDEXEjira_export"]
        if configCollection.keyExists("Jira_user"):
            self.Jira_user = configCollection["Jira_user"]

        if configCollection.keyExists("stripWhiteSpace"):
            self.stripWhiteSpace = configCollection["stripWhiteSpace"]
        if configCollection.keyExists("convertToLower"):
            self.convertToLower = configCollection["convertToLower"]
        if configCollection.keyExists("convertToASCII"):
            self.convertToASCII = configCollection["convertToASCII"]
        if configCollection.keyExists("singleSpaces"):
            self.singleSpaces = configCollection["singleSpaces"]

        if configCollection.keyExists("loadDocument"):
            self.loadDocument = configCollection["loadDocument"]
        if configCollection.keyExists("rawTextFromDocument"):
            self.rawTextFromDocument = configCollection["rawTextFromDocument"]
        if configCollection.keyExists("finalJSONfromRaw"):
            self.finalJSONfromRaw = configCollection["finalJSONfromRaw"]
        if configCollection.keyExists("prepareBM25corpus"):
            self.prepareBM25corpus = configCollection["prepareBM25corpus"]
        if configCollection.keyExists("completeBM25database"):
            self.completeBM25database = configCollection["completeBM25database"]
        if configCollection.keyExists("vectorizeFinalJSON"):
            self.vectorizeFinalJSON = configCollection["vectorizeFinalJSON"]

        if configCollection.keyExists("issuePattern"):
            self.issuePattern = configCollection["issuePattern"]
        if configCollection.keyExists("issueTemplate"): 
            self.issueTemplate = configCollection["issueTemplate"]
        if configCollection.keyExists("extractPattern"): 
            self.extractPattern = configCollection["extractPattern"]
        if configCollection.keyExists("assignList"): 
            self.assignList = configCollection["assignList"]

        # manually call model validator
        self.indexerWorkflow_verify_configuration()


    def processText(self, textIn : str) -> str:
        """
        Process text per flags (strip, lower, convert to ASCII, single space)
        
        :param textIn: text to process
        :type textIn: str
        :return: processed text
        :rtype: str
        """
        if self.stripWhiteSpace:
            textIn = textIn.strip()
        if self.convertToLower:
            textIn = textIn.lower()
        if self.convertToASCII:
            textIn = anyascii(textIn)
        if self.singleSpaces:
            textIn = " ".join(textIn.split())
        return textIn


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
            pageContent = self.processText(pageContent)
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
        
        docs = self.processText(docs)
        return docs


    def loadDocumentPhase(self) -> int: 
        """
        Load document and store it as plain text

        :return: Length of extracted text 
        :rtype: int
        """

        textCombined = self.loadPDFPyPDFLoader(self.inputFileName)
        if not textCombined:
            textCombined = self.loadPDFpymupdf4llm(self.inputFileName)

        with open(self.rawTextFromDoc, "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)

        return len(textCombined)


    def preprocessReportRawTextPhase(self, rawText : str) -> dict[str, str] :
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

        compiledPattern = re.compile(self.issuePattern)
        start = -1
        end = -1
        dictIssues = {}
        uniqIdx = 0
        prevMatch = None

        for match in re.finditer(compiledPattern, rawText) :

            end = match.start()

#            if prevMatch and prevMatch.group("IDENTIFIERGROUP"):
#                print(f"match: {prevMatch.start()}, IDENTIFIERGROUP: {prevMatch.group('IDENTIFIERGROUP')} start: {start}   end:{end}")

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
        compiledExtract = re.compile(self.extractPattern)
        match = re.search(compiledExtract, docs)
        if match:
            oneIssue = ClassTemplate()
            for i in range(len(self.assignList)) :
                attrName = self.assignList[i]
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

        ollModel = self.createOpenAIModel()

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


    def parseAllIssuesPhase(self, listText: Dict[str, str], ClassTemplate : BaseModel) -> RecordCollection :
        """
        Extracts ClassTemplate instances in the dict of pages using LLM. ClassTemplate is based on Pydantic BaseModel.
        
        Args:
            dictText (Dict[str, str]) - dict of pages
            ClassTemplate (BaseModel) - issue template

        Returns:
            RecordCollection
        """

        recordCollection = RecordCollection(
            report = str(Path(self.inputFileName).name),
            finding_dict = {}
        )

        totalUsage = RunUsage()
        totalStartTime = time.time()

        for item in listText.keys():
            startOneIssue = time.time()

            oneIssue, usageStats = self.parseIssueOllama(listText[item], ClassTemplate)
            if not oneIssue:
                continue

            recordCollection[item] = oneIssue

            endOneIssue = time.time()
            if usageStats:
                msg = f"Record: <b>{item}</b>. Usage: {self.usageFormat(usageStats)}. Time: {(endOneIssue - startOneIssue):9.2f} seconds."
                self.addUsage(usageStats)
                totalUsage += usageStats
            else:
                msg = f"Record: <b>{item}</b>. {(endOneIssue - startOneIssue):9.2f} seconds."
                self.workerSnapshot(msg)

        totalEndTime = time.time()
        msg = f"Total usage: {self.usageFormat(totalUsage)}. Total time {(totalEndTime - totalStartTime):9.2f} seconds."
        self.workerSnapshot(msg)

        return recordCollection

    
    def jiraExportPhase(self, ClassTemplate : BaseModel) -> RecordCollection :
        """
        Export issues from Jira project
        Transform to smaller records
        Write as final JSON for vectorization

        Args:
            ClassTemplate (BaseModel) - issue template

        Returns:
            RecordCollection - all items
        """

        # Connect to Jira
        try:
            jira = JIRA(server=self.INDEXEjira_url, basic_auth=(self.jira_user, self.jira_api_token))
        except Exception as e:
            msg = f"Jira API exception: {e}"
            self.workerError(msg)
            return 0
        if not jira:
            msg = f"Jira REST API connection error"
            self.workerError(msg)
            return 0

        jql_query = f'project = {self.inputFileName}'
        recordCollection = RecordCollection(finding_dict = {})

        # Fetch issues from Jira
        # default maxResults is 50, we need more than that
        issues = jira.search_issues(jql_query, maxResults=self.INDEXEjira_max_results, json_result = True)
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


    def bm25sAddReportToCorpusPhase(self, corpus : list[str], issues: RecordCollection, ClassTemplate : BaseModel) -> List[str] :
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


    def bm25sProcessCorpusPhase(self, corpus : list[str], folderName: str) -> List[List[str]] :
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
#        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(folderName)
        return corpus_tokens


    def vectorizeFinalJSONPhase(self, recordCollection : RecordCollection, ClassTemplate : BaseModel) -> tuple[int, int] :
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

        accepted = 0
        rejected = 0

        if not self.initRAGcomponents():
            return accepted, rejected

        if self.INDEXEjira_export :
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

        for key in recordCollection.finding_dict:
            reportItem = ClassTemplate.model_validate(recordCollection[key])

            inputHashFunc = hashlib.sha256()
            inputStringToHash = reportItem.stringToHash()
            inputHashFunc.update(inputStringToHash.encode('utf-8'))
            inputRecordHash = inputHashFunc.hexdigest()

            uniqueId = key
            queryResult = chromaCollection.get(ids=[uniqueId])
            if (len(queryResult["ids"])) :

                existingRecord = ClassTemplate.model_validate_json(queryResult["documents"][0])

                existingHashFunc = hashlib.sha256()
                existingStringToHash = existingRecord.stringToHash()
                existingHashFunc.update(existingStringToHash.encode('utf-8'))
                existingHash = existingHashFunc.hexdigest()

                if inputRecordHash == existingHash:
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

#            if self.INDEXEjira_export :
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

        with open(self.finalJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(recordCollection.model_dump_json(indent=2))

        return

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

        msg = f"Document: <b>{self.inputFileBaseName}</b>"
        self.workerSnapshot(msg)
        msg = f"Raw text file: <b>{self.rawTextFromDoc}</b>"
        self.workerSnapshot(msg)
        msg = f"Raw JSON file: <b>{self.rawJSON}</b>"
        self.workerSnapshot(msg)
        msg = f"Final JSON file: <b>{self.finalJSON}</b>"
        self.workerSnapshot(msg)

        # ---------------stage Jira export
        if self.INDEXEjira_export :

            startTime = totalStart

            recordCollection = self.jiraExportPhase(issueTemplate)
            accepted, rejected = self.vectorize(recordCollection, issueTemplate)

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted}  rejected {rejected}."
            self.workerSnapshot(msg)

            totalEnd = time.time()
            msg = f"Processing completed. Total time {(totalEnd - totalStart):9.2f} seconds."
            self.workerSnapshot(msg)
            return

        # ---------------stage read pdf ---------------
        if self.loadDocument :
            startTime = totalStart

            textCombined = self.loadDocumentPhase(self.inputFileName)
            with open(self.rawTextFromDoc, "w" , encoding="utf-8", errors="ignore") as rawOut:
                rawOut.write(textCombined)
            endTime = time.time()

            inputFileBaseName = str(Path(self.inputFileName).name)
            msg = f"Read input document {self.inputFileBaseName}. Time: {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage preprocess raw text ---------------
        if self.rawTextFromDocument :
            startTime = time.time()

            if 'textCombined' not in locals():
                # if this is a separate step - read extracted text file
                with open(self.rawTextFromDoc, "r", encoding='utf8', errors='ignore') as txtIn:
                    textCombined = txtIn.read()
                msg = f"Read raw text from file {self.inputFileBaseName}."
                self.workerSnapshot(msg)

            dictRawIssues = self.preprocessReportRawTextPhase(textCombined)
            with open(self.rawJSON, "w", encoding="utf-8", errors="ignore") as jsonOut:
                jsonOut.writelines(json.dumps(dictRawIssues, indent=2))
            endTime = time.time()

            rawTextFromDocBaseName = str(Path(self.rawTextFromDoc).name)
            msg = f"Preprocessed raw text {rawTextFromDocBaseName}. Found {len(dictRawIssues)} potential issues. Time: {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage create final JSON ---------------
        if self.finalJSONfromRaw :
            startTime = time.time()

            if 'dictRawIssues' not in locals():
                # if this is a separate step - read raw JSON into record collection
                with open(self.rawJSON, "r", encoding='utf8', errors='ignore') as jsonIn:
                    dictRawIssues = json.load(jsonIn)
                msg = f"Read {len(dictRawIssues)} raw records from file {self.inputFileBaseName}."
                self.workerSnapshot(msg)

            recordCollection = self.parseAllIssuesPhase(self.inputFileName, dictRawIssues, issueTemplate)
            self.writeFinalJSON(recordCollection)

            endTime = time.time()

            finalJSONBaseName = str(Path(self.finalJSON).name)
            msg = f"Found {recordCollection.objectCount()} records. Wrote final JSON: <b>{finalJSONBaseName}</b>. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage bm25s preparation ---------------
        if self.prepareBM25corpus :

            startTime = time.time()

            if 'recordCollection' not in locals():
                # if this is a separate step - read final JSON into record collection
                with open(self.finalJSON, "r", encoding='utf8', errors='ignore') as txtIn:
                    textStr = txtIn.read()
                recordCollection = RecordCollection.model_validate_json(textStr)
                msg = f"Read {recordCollection.objectCount()} records from file {self.inputFileBaseName}."
                self.workerSnapshot(msg)

            corpus = self.bm25sAddReportToCorpusPhase(corpus, recordCollection, issueTemplate)

            endTime = time.time()

            msg = f"Added {recordCollection.objectCount()} records to BM25 corpus. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)

        # ---------------stage bm25s completion ---------------
        if self.completeBM25database :

            startTime = time.time()

            folderName = self.bm25IndexFolder
            self.bm25sProcessCorpusPhase(corpus, folderName)

            endTime = time.time()

            msg = f"Created BM25 database in {folderName}. {(endTime - startTime):9.2f} seconds"
            self.workerSnapshot(msg)


        # ---------------stage vectorizeFinalJSON --------------
        if self.vectorizeFinalJSON :

            startTime = time.time()

            if 'recordCollection' not in locals():
                # if this is a separate step - read final JSON into record collection
                with open(self.finalJSON, "r", encoding='utf8', errors='ignore') as txtIn:
                    textStr = txtIn.read()
                recordCollection = RecordCollection.model_validate_json(textStr)
                self.workerSnapshot(msg)

            accepted, rejected = self.vectorizeFinalJSONPhase(recordCollection, issueTemplate)

            endTime = time.time()
            msg = f"Processed {recordCollection.objectCount()}, accepted {accepted} rejected {rejected}."
            self.workerSnapshot(msg)

        # ---------------stage completed ---------------

        totalEnd = time.time()
        msg = f"Processing completed. Usage: {self.totalUsageFormat()}. Total time {(totalEnd - totalStart):9.2f} seconds."
        self.workerSnapshot(msg)
