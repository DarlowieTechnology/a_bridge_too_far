#
# Indexer workflow class used by Django app and command line
#
from typing import List, Dict, Any
from typing_extensions import Self
import threading
import json
import re
import time
from pathlib import Path
import hashlib
from pprint import pprint

from pydantic import BaseModel, ValidationError, Field, model_validator
import pydantic_ai
from pydantic_ai import Agent, AgentRunResult
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.usage import RunUsage

import chromadb
from chromadb import Collection, ClientAPI
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from langchain_community.document_loaders.pdf import PyPDFLoader

import pymupdf.layout  # activate PyMuPDF-Layout in pymupdf
import pymupdf4llm

import Stemmer
import bm25s

from anyascii import anyascii


# local
from common import COLLECTION, RecordCollection, ConfigCollection, DebugUtils, OpenFile
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory

class IndexerWorkflow(WorkflowBase):

    session_key : str = Field(default = "", description="logger session name for Indexer CLI")

    globalDataFolder : str = Field(default = "", description="Global data folder")
    documentFolder : str = Field(default = "", description="Source document folder")
    dataFolder : str = Field(default = "", description="INDEXER interim data folder")

    # file names : Original -> Raw Text -> Raw JSON -> Final JSON
    #
    inputFileName : str = Field(default = "", description="Original document name")
    interimFolder : str = Field(default = "", description="Full interim folder path")
    rawTextFromDoc : str = Field(default = "", description="Raw text name")
    rawJSON : str = Field(default = "", description="Raw JSON name")
    finalJSON : str = Field(default = "", description="Final JSON name")

    inputFileBaseName : str = Field(default = "", description="Base name of original document")

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
    dictDocuments : Dict[str, Any] = Field(default = [], description="Template descriptions")

    issuePattern : str = Field(default = "", description="issue regexp pattern")
    issueTemplateName : str = Field(default = "", description="issue template name")
    extractPattern : str = Field(default = "", description="issue field extract regexp pattern")
    assignList : List[str] = Field(default = [], description="list of issue fields to assign")

    corpus : List[str] = Field(default = [], description="corpus for bm25s")


    @model_validator(mode='after')
    def indexerWorkflow_verify_configuration(self) -> Self:

        if not Path(self.interimFolder).is_dir:
            raise ValueError(f'Full interim data folder path is invalid')

        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        self.globalDataFolder = configCollection["GLOBALdataFolder"]
        self.documentFolder = configCollection["INDEXEdocumentFolder"]
        self.dataFolder = configCollection["INDEXEdataFolder"]
        self.interimFolder = self.globalDataFolder + self.documentFolder + self.dataFolder

        # make interim data folder if does not exist
        Path(self.interimFolder).mkdir(parents=True, exist_ok=True)

        # template description
        documentJSONName = configCollection["GLOBALdataFolder"] + configCollection["INDEXEdocumentFolder"] + "documents.json"
        result, fileContentOrError = OpenFile.open(filePath = documentJSONName, readContent = True)
        if not result:
            raise ValueError(f'Cannot read template description file')
        else:
            self.dictDocuments = json.loads(fileContentOrError)


        if configCollection.keyExists("session_key"):
            self.session_key = configCollection["session_key"]

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


    def loadDocumentPhase(self): 
        """
        Load document and store it as plain text
        """

        startTime = time.time()

        textCombined = self.loadPDFPyPDFLoader(self.inputFileName)
        if not textCombined:
            textCombined = self.loadPDFpymupdf4llm(self.inputFileName)

        with open(self.rawTextFromDoc, "w" , encoding="utf-8", errors="ignore") as rawOut:
            rawOut.write(textCombined)

        self.updateStats(topKey = "Load Document", keyValList = [("Time", time.time() - startTime), ("Files", 1), ("Length", len(textCombined)), ("PDF", 1), ("PDF length", len(textCombined))])


    def loadDocumentPhaseAllFiles(self, inputFileList : List[str]): 
        """
        Load all documents and store as plain text

        Args:
            inputFileList (List[str]) - list of files
        
        Returns:

        """
        for inputFileName in inputFileList:
            self.inputFileName = self.globalDataFolder + self.documentFolder + inputFileName
            self.rawTextFromDoc = self.interimFolder + inputFileName + ".raw.txt"
            self.loadDocumentPhase()

#            context["rawTextFromDoc"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.txt"
#            context["rawJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.json"
#            context["finalJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".json"

            # base input file name is a key for template dict
#            self.inputFileBaseName = str(Path(inputFileName).name)
           
            # set raw text parsing configuration
#            self.issuePattern = self.dictDocuments[self.inputFileBaseName]["pattern"]
#            self.issueTemplateName = self.dictDocuments[self.inputFileBaseName]["templateName"]
#            self.extractPattern = self.dictDocuments[self.inputFileBaseName]["extract"]
#            self.assignList = self.dictDocuments[self.inputFileBaseName]["assign"]



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
        match : Any = None

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
        if match:
            if match.group("TERMINATORGROUP"):

                self.updateStats(topKey = "Raw JSON", keyValList = [ ("Raw item files", 1), ("Raw items", len(dictIssues)) ])
                return dictIssues    

        end = len(rawText)
        if prevMatch:
            dictIssues[prevMatch.group(0)] = rawText[start:end]

        self.updateStats(topKey = "Raw JSON", keyValList = [ ("Raw item files", 1), ("Raw items", len(dictIssues)) ])
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

        compiledExtract = re.compile(self.extractPattern)
        match = re.search(compiledExtract, docs)
        if match:
            oneIssue = ClassTemplate()
            for i in range(len(self.assignList)) :
                attrName = self.assignList[i]
                setattr(oneIssue, attrName, match.group(attrName))

            self.updateStats(topKey = "Parse", keyValList = [ ("Issues parsed with fallback regexp", 1) ])

            return oneIssue
        else:
            self.updateStats(topKey = "Parse", keyValList = [ ("Issues failed to parse", 1) ])
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

            result : AgentRunResult = agent.run_sync(prompt)
            oneIssue = ClassTemplate.model_validate_json(result.output.model_dump_json())
            for attr in oneIssue.__dict__:
                if oneIssue.__dict__[attr]:
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
            runUsage : RunUsage = result.usage()

            self.updateStats(topKey = "Parse", keyValList = [ ("Issues parsed with LLM", 1) ])

            return oneIssue, runUsage
        
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)

        # attempt regexp match only if LLM match failed
        return self.parseFallback(docs, ClassTemplate), None


    def parseAllIssuesPhase(self, listText: Dict[str, str], ClassTemplate : BaseModel) -> tuple[ RecordCollection, RunUsage ] :
        """
        Extract ClassTemplate instances using LLM. Fallback on regexp.
        
        Args:
            dictText (Dict[str, str]) - dict of pages
            ClassTemplate (BaseModel) - issue template

        Returns:
            tuple of RecordCollection and RunUsage
        """

        recordCollection = RecordCollection(
            report = str(Path(self.inputFileName).name),
            finding_dict = {}
        )

        usageForPhase = RunUsage()

        for item in listText.keys():

            oneIssue, usageStats = self.parseIssueOllama(listText[item], ClassTemplate)
            if not oneIssue:
                continue

            recordCollection[item] = oneIssue

            if usageStats:
                self.addUsage(usageStats)
                usageForPhase += usageStats

        return recordCollection, usageForPhase

    


    @staticmethod
    def bm25sProcessCorpusPhase(corpus : list[str], folderName: str) -> List[List[str]] :
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


    def vectorizeFinalJSONPhase(self, ClassTemplate : BaseModel) :
        """
        Add all structured records to vector database.
        Before vectorization improve English text
        1. Lowercase
        2. Drop stop words
       
        Args:
            ClassTemplate (BaseModel) - issue template

        """

        startTime = time.time()

        # read final JSON into record collection
        result, fileContentOrError = OpenFile.open(filePath = self.finalJSON, readContent = True)
        if not result:
            msg = f"vectorizeFinalJSONPhase: {fileContentOrError} - perform 'parseIssues' action first"
            self.workerSnapshot(msg)
            return
        else:
            recordCollection = RecordCollection.model_validate_json(fileContentOrError)

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

        self.updateStats(topKey = "Vectorize", keyValList = [ ("Time", time.time() - startTime),  ("Accepted", accepted), ("Rejected", rejected) ])


    def rawTextFromDocumentPhase(self) :
        """
        Perform conversion of raw text to raw JSON

        Args:
        
        Returns:
        """

        startTime = time.time()

        result, fileContentOrError = OpenFile.open(filePath = self.rawTextFromDoc, readContent = True)
        if not result:
            msg = f"preprocess: {fileContentOrError} - perform 'loadDocument' phase first"
            print(msg)
            return
        else:
            textCombined = fileContentOrError

        dictIssues = self.preprocessReportRawTextPhase(textCombined)

        with open(self.rawJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(json.dumps(dictIssues, indent=2))

        self.updateStats(topKey = "Raw JSON", keyValList = [ ("Time", time.time() - startTime),  ("Records", len(dictIssues)) ])


    def rawTextFromDocumentPhaseAllFiles(self, inputFileList : List[str]):
        """
        Perform conversion of raw text to raw JSON for all files


        Args:
            inputFileList (List[str]) - list of files
        
        Returns:
        """
        for inputFileName in inputFileList:
            self.inputFileName = self.globalDataFolder + self.documentFolder + inputFileName
            self.inputFileBaseName = str(Path(inputFileName).name)
            self.rawTextFromDoc = self.interimFolder + inputFileName + ".raw.txt"
            self.rawJSON = self.interimFolder + inputFileName + ".raw.json"

            # set raw text parsing configuration
            self.issuePattern = self.dictDocuments[self.inputFileBaseName]["pattern"]
            self.issueTemplateName = self.dictDocuments[self.inputFileBaseName]["templateName"]
            self.extractPattern = self.dictDocuments[self.inputFileBaseName]["extract"]
            self.assignList = self.dictDocuments[self.inputFileBaseName]["assign"]

            self.rawTextFromDocumentPhase()
            print(self.showStats(topKey = "Raw JSON", showKey = "Raw item files", label="Number of files processed"))



    def finalJSONfromRawPhase(self, issueTemplate : BaseModel) :
        """
        Perform conversion of raw JSON to final JSON

        Args:
            issueTemplate (BaseModel) - description of structured data object
        
        Returns:
        """

        startTime = time.time()

        # read raw JSON into dict
        result, fileContentOrError = OpenFile.open(filePath = self.rawJSON, readContent = True)
        if not result:
            msg = f"finalJSONfromRawPhase: {fileContentOrError} - perform 'rawTextFromDocument' phase first"
            self.workerSnapshot(msg)
            return
        else:
            dictRawIssues : Dict[str, str] = json.loads(fileContentOrError)

        recordCollection, totalUsage = self.parseAllIssuesPhase(dictRawIssues, issueTemplate)
        with open(self.finalJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(recordCollection.model_dump_json(indent=2))

        self.updateStats(topKey = "Final JSON", keyValList = [ ("Time", time.time() - startTime),  ("Records", recordCollection.objectCount()), ("Input Tokens", totalUsage.input_tokens), ("Output Tokens", totalUsage.output_tokens) ])


    def prepareBM25corpusPhase(self, issueTemplate : BaseModel, corpus : List[str]) -> List[str] :
        """
        Add issues from one document to BM25s corpus

        Args:
            issueTemplate (BaseModel) - description of structured data object
            corpus (List[str]) - common corpus
        
        Returns:
            corpus (List[str]) - common corpus
        """

        startTime = time.time()

        # read final JSON into record collection
        result, fileContentOrError = OpenFile.open(filePath = self.finalJSON, readContent = True)
        if not result:
            msg = f"prepareBM25corpusPhase: {fileContentOrError} - perform 'finalJSONfromRaw' phase first"
            self.workerSnapshot(msg)
            return
        else:
            recordCollection = RecordCollection.model_validate_json(fileContentOrError)

        for key in recordCollection.finding_dict:
            issue = recordCollection.finding_dict[key]
            reportItem = issueTemplate.model_validate(issue)
            issueText = reportItem.bm25s()
            corpus.append(issueText)

        self.updateStats(topKey = "Prepare BM25 Corpus", keyValList = [ ("Time", time.time() - startTime) ])
        return corpus


    def threadWorker(self, issueTemplate : BaseModel, corpus : List[str]) :

        totalStart = time.time()

        msg = f"Document: {self.inputFileBaseName}"
        self.workerSnapshot(msg)
        msg = f"Interim folder: {self.interimFolder}"
        self.workerSnapshot(msg)
        msg = f"Raw text file: {self.rawTextFromDoc}"
        self.workerSnapshot(msg)
        msg = f"Raw JSON file: {self.rawJSON}"
        self.workerSnapshot(msg)
        msg = f"Final JSON file: {self.finalJSON}"
        self.workerSnapshot(msg)

        # ---------------loadDocument phase ---------------
        if self.loadDocument :
            self.loadDocumentPhase()

        # ---------------phase rawTextFromDocument ---------------
        if self.rawTextFromDocument :
            self.rawTextFromDocumentPhase()

        # ---------------phase finalJSONfromRaw ---------------

        if self.finalJSONfromRaw :
            self.finalJSONfromRawPhase(issueTemplate = issueTemplate)

        # ---------------phase prepareBM25corpus ---------------
        if self.prepareBM25corpus :
            corpus = self.prepareBM25corpusPhase(issueTemplate = issueTemplate, corpus = corpus)

        # ---------------stage vectorizeFinalJSON --------------
        if self.vectorizeFinalJSON :
            self.vectorizeFinalJSONPhase(issueTemplate)

        # ---------------stage completed ---------------

        self.updateStats(topKey = "Total", keyValList = [ ("Time", time.time() - totalStart), ("Usage", self.totalUsageFormat(insertHTML = False) ) ])

        pprint(self.stats)


    @staticmethod
    def threadWorkerStatic(context : Dict[str, Any], fileList : List[str]):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            context (Dict[str, Any]) - configuration dict
            fileList (List[str]) - list of file to process
        
        Returns:
            None
        """

        # bm25s index is common for all source documents
        corpus : List[str] = []

        for fileName in fileList:

            context["inputFileBaseName"] = fileName
            context["interimFolder"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"]
            context["inputFileName"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + fileName
            context["rawTextFromDoc"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.txt"
            context["rawJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".raw.json"
            context["finalJSON"] = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEdataFolder"] + fileName + ".json"

            context["statusFileName"] = context["IDXCLIstatus_FileName"]
            context["session_key"] = context["IDXCLIsession_key"]

            configCollection = ConfigCollection(context)
            indexerWorkflow = IndexerWorkflow()
            indexerWorkflow.configure(configCollection)

            issueTemplate = ParserClassFactory.factory(indexerWorkflow.issueTemplateName)

            thread = threading.Thread( target=indexerWorkflow.threadWorker, args=(issueTemplate, corpus))
            thread.start()
            thread.join()

        # create common bm25s index for all source documents
        if context["completeBM25database"] :
            folderName = context["GLOBALdataFolder"] + context["INDEXEdocumentFolder"] + context["INDEXEbm25IndexFolder"]
            IndexerWorkflow.bm25sProcessCorpusPhase(corpus=corpus, folderName = folderName)
