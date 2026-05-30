#
# Indexer workflow class used by Django app and command line
#
from typing import List, Dict, Any
from typing_extensions import Self
import sys
import logging
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
from common import COLLECTION, CommonHelper, RecordCollection, ConfigCollection, DebugUtils, OpenFile
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory

class IndexerWorkflow(WorkflowBase):

    GLOBALllm_Provider : str = Field(default = "", description="Global provider of LLM service")
    GLOBALllm_Embed : str = Field(default = "", description="Embedding LLM")
    GLOBALembedding_URL : str = Field(default = "", description="Embedding LLM")
    GLOBALllm_Version : str = Field(default = "", strict=True, description="General LLM")
    GLOBALllm_URL : str = Field(default = "", description="Global LLM service base URL")

    logginglevel : int = Field(default = logging.WARN, description="Logging level")

    statusFileName : str = Field(default = "INDEXERLOG", description="Name of status log file")
    ragDatapath : str = Field(default = "chromadb", description="Path to RAG database")
    documentFolder : str = Field(default = "", description="Source document folder")
    dataFolder : str = Field(default = "", description="interim data folder")
    bm25IndexFolder: str = Field(default = "", description="bm25s index folder path")
    templateJSONName : str = Field(default = "", description="Document template filename")

    # file names : Original -> Raw Text -> Raw JSON -> Final JSON
    #
    inputFileName : str = Field(default = "", description="Original document name")
    rawTextFromDoc : str = Field(default = "", description="Raw text name")
    rawJSON : str = Field(default = "", description="Raw JSON name")
    finalJSON : str = Field(default = "", description="Final JSON name")

    inputFileBaseName : str = Field(default = "", description="Base name of original document")

    # text processing flags
    stripWhiteSpace : bool = Field(default = True, description="Strip excessive whitespace characters from source text")
    convertToLower : bool = Field(default = True, description="Covert all characters in source text to lowercase")
    convertToASCII : bool = Field(default = True, description="Covert all characters in source text to ASCII")
    singleSpaces : bool = Field(default = True, description="Replace multiple space characters with single space in source text")

    # workflow actions
    loadDocument : bool = Field(default = True, description="Load text from source documents")
    rawTextFromDocument : bool = Field(default = True, description="preprocess text from source documents")
    finalJSONfromRaw : bool = Field(default = True, description="create final JSON")
    prepareBM25corpus : bool = Field(default = True, description="prepare BM25s corpus")
    vectorizeFinalJSON : bool = Field(default = True, description="vectorize final JSON")
    clear : bool = Field(default = False, description="Clear intermediate files")

    issuePattern : str = Field(default = "", description="issue regexp pattern")
    issueTemplateName : str = Field(default = "", description="issue template name")
    extractPattern : str = Field(default = "", description="issue field extract regexp pattern")
    assignList : List[str] = Field(default = [], description="list of issue fields to assign")


    @model_validator(mode='after')
    def verify_configuration(self) -> Self:

        if not Path(self.dataFolder).is_dir:
            raise ValueError(f'Data folder path is invalid')
        
        if not Path(self.bm25IndexFolder).is_dir:
            raise ValueError(f'BM25s index folder path is invalid')
        return self


    def configure(self, configCollection : ConfigCollection) :

        # call base class configuration first
        super().configure(configCollection)

        self.GLOBALllm_Provider = configCollection["GLOBALllm_Provider"]
        self.GLOBALllm_Embed = configCollection["GLOBALllm_Embed"]
        self.GLOBALembedding_URL = configCollection["GLOBALembedding_URL"]
        self.GLOBALllm_Version = configCollection["GLOBALllm_Version"]
        self.GLOBALllm_URL = configCollection["GLOBALllm_URL"]

        if configCollection.keyExists("logginglevel"):
            self.logginglevel = configCollection["logginglevel"]
        logging.basicConfig(stream=sys.stdout, level=self.logginglevel)

        self.logger = logging.getLogger(configCollection["GLOBALloggerSessionKey"])

        if configCollection.keyExists("statusFileName"):
            self.statusFileName = configCollection["statusFileName"]

        if configCollection.keyExists("ragDatapath"):
            # WEB update, CLI advanced settings
            self.ragDatapath = configCollection["ragDatapath"]
        else:
            # CLI and WEB init
            self.ragDatapath = configCollection["INDEXERAGFolder"]

        if configCollection.keyExists("documentFolder"):
            # WEB update, CLI advanced settings
            self.documentFolder = configCollection["documentFolder"]
        else:
            # CLI and WEB init
            self.documentFolder = configCollection["INDEXEdocumentFolder"]

        if configCollection.keyExists("dataFolder"):
            # WEB update, CLI advanced settings
            self.dataFolder = configCollection["dataFolder"]
        else:
            # CLI and WEB init
            self.dataFolder = configCollection["INDEXEdataFolder"]

        if configCollection.keyExists("bm25IndexFolder"):
            # WEB update, CLI advanced settings
            self.bm25IndexFolder = configCollection["bm25IndexFolder"]
        else:
            # CLI and WEB init
            self.bm25IndexFolder = configCollection["INDEXEbm25IndexFolder"]

        # make bm25s index folder if does not exist
        Path(self.bm25IndexFolder).mkdir(parents=True, exist_ok=True)

        # make data folder if does not exist
        Path(self.dataFolder).mkdir(parents=True, exist_ok=True)

        # template description
        self.templateJSONName = self.documentFolder + "documents.json"

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
        if configCollection.keyExists("vectorizeFinalJSON"):
            self.vectorizeFinalJSON = configCollection["vectorizeFinalJSON"]
        if configCollection.keyExists("clear"):
            self.clear = configCollection["clear"]

        # manually call model validator
        self.verify_configuration()


    def openTemplateFile(self, templateName : str) -> BaseModel :
        """
        Open template definition file and read template for the document.
        Raise exception on error.
        
        :param templateName: filename as key in template dict
        :type templateName: str
        :return: Object based on BaseModel
        :rtype: BaseModel
        """

        dictDocuments : Dict[str, Dict] = {}
        result, fileContentOrError = OpenFile.open(filePath = self.templateJSONName, readContent = True)
        if not result:
            raise ValueError(f'Cannot read template description file')
        else:
            dictDocuments = json.loads(fileContentOrError)

        if templateName not in dictDocuments.keys():
            raise ValueError(f'Cannot read template description')

        # set raw text parsing configuration
        self.issuePattern = dictDocuments[templateName]["pattern"]
        self.issueTemplateName = dictDocuments[templateName]["templateName"]
        self.extractPattern = dictDocuments[templateName]["extract"]
        self.assignList = dictDocuments[templateName]["assign"]

        templateInstance = ParserClassFactory.factory(self.issueTemplateName)
        return templateInstance


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
            return None       
        
        docs = self.processText(docs)
        return docs


    def loadDocumentPhase(self): 
        """
        Load document and store it as plain text

        Args:
        
        Returns:

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
            self.updateStats(topKey = "Load Document", keyValList = [ ("Load Document", inputFileName)])
            self.inputFileName = self.documentFolder + inputFileName
            self.rawTextFromDoc = self.dataFolder + inputFileName + ".raw.txt"
            self.loadDocumentPhase()
            print(self.showStats(topKey = "Load Document", showKey = "Load Document", label="Loaded"))
        self.removeStats(topKey = "Load Document", removeKey = "Load Document")


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
        else:
            msg = f"No matches found. Potential invalid regexp for document {self.inputFileName}"
            self.workerSnapshot(msg)

        end = len(rawText)
        if prevMatch:
            dictIssues[prevMatch.group('IDENTIFIERGROUP')] = rawText[start:end]

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

            self.updateStats(topKey = "Final JSON", keyValList = [ ("Issues parsed with fallback regexp", 1) ])

            return oneIssue
        else:
            self.updateStats(topKey = "Final JSON", keyValList = [ ("Issues failed to parse", 1) ])
            return None


    def parseIssueOllama(self, ident : str, docs : str, ClassTemplate : BaseModel) -> tuple[BaseModel, RunUsage] :
        """
        Use Ollama host and Pydantic AI Agent to extracts one ClassTemplate structured record. 
        ClassTemplate is a subclass of Pydantic BaseModel.
        
        Args:
            ident (str) - identifier extracted via regexp
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        """

        systemPrompt = f"""
        The prompt contains a record.
        Record starts with identifier field with the value: {ident}
        Here is the JSON schema for the ClassTemplate model you must use as context for what information is expected:
        {json.dumps(ClassTemplate.model_json_schema(), indent=2)}
        """

#        print("======SYS======")
#        print(systemPrompt)
#        print("============")

        prompt = f"{docs}"

#        print("======INPUT======")
#        print(docs)
#        print("============")

        ollModel = self.createOpenAIModel()

        agent = Agent(ollModel,
                    output_type=ClassTemplate,
                    system_prompt = systemPrompt,
                    retries=5)
        try:
            result : AgentRunResult = agent.run_sync(prompt)

            if not result.output.identifier:
                # cannot parse identifier - replacing with identifier from regexp before validation
                result.output.identifier = ident

#            print("======Output======")
#            print(result.output)
#            print("============")

            oneIssue = ClassTemplate.model_validate_json(result.output.model_dump_json())

            for attr in oneIssue.__dict__:
                if oneIssue.__dict__[attr]:
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].replace("\n", " ")
                    oneIssue.__dict__[attr] = oneIssue.__dict__[attr].encode("ascii", "ignore").decode("ascii")
            runUsage : RunUsage = result.usage()

            self.updateStats(topKey = "Final JSON", keyValList = [ ("Issues parsed with LLM", 1) ])

            return oneIssue, runUsage
        
        except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
            msg = "Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerSnapshot(msg)
            print(f"ident: {ident}  exception: {e}")
        except ValidationError as e:
            msg = f"Exception: ValidationError {e}"
            self.workerSnapshot(msg)
            print(f"ident: {ident}  exception: {e}")

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
#            time.sleep(7)
            oneIssue, usageStats = self.parseIssueOllama(item, listText[item], ClassTemplate)
            if not oneIssue:
                continue

            recordCollection[item] = oneIssue

            if usageStats:
                self.addUsage(usageStats)
                usageForPhase += usageStats

        return recordCollection, usageForPhase

    
    def vectorizeFinalJSONPhase(self, classTemplate : BaseModel) :
        """
        Add all structured records to vector database.
        Before vectorization improve English text
        1. Lowercase
        2. Drop stop words
       
        Args:
            classTemplate (BaseModel) - issue template

        """

        startTime = time.time()

        # read final JSON into record collection

        result, fileContentOrError = OpenFile.open(filePath = self.finalJSON, readContent = True)
        if not result:
            msg = f"vectorizeFinalJSONPhase: {fileContentOrError} - perform 'Final JSON' action first"
            self.workerSnapshot(msg)
            return
        else:
            recordCollection = RecordCollection.model_validate_json(fileContentOrError)

        accepted = 0
        rejected = 0

        if not self.initRAGcomponents():
            return accepted, rejected

        chromaCollection = self.openOrCreateCollection(collectionName = COLLECTION.ISSUES.value, createFlag = True)
        if not chromaCollection:
            return 0, 0

        ids : list[str] = []
        docs : list[str] = []
        docMetadata : list[str] = []
        embeddings = []

        for key in recordCollection.finding_dict:
            reportItem = classTemplate.model_validate(recordCollection[key])

            inputHashFunc = hashlib.sha256()
            inputStringToHash = reportItem.stringToHash()
            inputHashFunc.update(inputStringToHash.encode('utf-8'))
            inputRecordHash = inputHashFunc.hexdigest()

            uniqueId = key
            queryResult = chromaCollection.get(ids=[uniqueId])
            if (len(queryResult["ids"])) :

                existingRecord = classTemplate.model_validate_json(queryResult["documents"][0])

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
        self.updateStats(topKey = "Vectorize", keyValList = [ ("Time", time.time() - startTime),  ("Vectors Accepted", accepted), ("Vectors Rejected", rejected) ])


    def vectorizeFinalJSONPhaseAllFiles(self, inputFileList : List[str]):
        """
        Add all structured records to vector database for all files

        Args:
            inputFileList (List[str]) - list of files
        
        Returns:
        """

        for inputFileName in inputFileList:
            self.inputFileName = self.documentFolder + inputFileName
            self.inputFileBaseName = str(Path(inputFileName).name)
            self.rawTextFromDoc = self.dataFolder + inputFileName + ".raw.txt"
            self.rawJSON = self.dataFolder + inputFileName + ".raw.json"
            self.finalJSON = self.dataFolder + inputFileName + ".json"

            templateInstance = self.openTemplateFile(self.inputFileBaseName)
            self.vectorizeFinalJSONPhase(templateInstance)


    def clearAllFiles(self):
        """
        Clear all intermediate files

        Args:
        
        Returns:
        """

        result, fileNameListOrError = OpenFile.readListOfFileNames(self.bm25IndexFolder, "*.*")
        if result:
            for fileName in fileNameListOrError:
                OpenFile.remove(fileName)


        result, fileNameListOrError = OpenFile.readListOfFileNames(self.bm25IndexFolder, "*.*")
        if result:
            for fileName in fileNameListOrError:
                OpenFile.remove(fileName)


    def rawTextFromDocumentPhase(self) :
        """
        Perform conversion of raw text to raw JSON

        Args:
        
        Returns:
        """
        startTime = time.time()

        result, fileContentOrError = OpenFile.open(filePath = self.rawTextFromDoc, readContent = True)
        if not result:
            msg = f"preprocess: {fileContentOrError} - perform '--load' phase first"
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
            self.updateStats(topKey = "Raw JSON", keyValList = [ ("Raw JSON", inputFileName)])
            self.inputFileName = self.documentFolder + inputFileName
            self.inputFileBaseName = str(Path(inputFileName).name)
            self.rawTextFromDoc = self.dataFolder + inputFileName + ".raw.txt"
            self.rawJSON = self.dataFolder + inputFileName + ".raw.json"

            self.openTemplateFile(self.inputFileBaseName)
            self.rawTextFromDocumentPhase()
            print(self.showStats(topKey = "Raw JSON", showKey = "Raw JSON", label="Raw JSON"))
        self.removeStats(topKey = "Raw JSON", removeKey = "Raw JSON")


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
            msg = f"finalJSONfromRawPhase: {fileContentOrError} - perform '--rawjson' phase first"
            self.workerSnapshot(msg)
            return
        else:
            dictRawIssues : Dict[str, str] = json.loads(fileContentOrError)

        recordCollection, totalUsage = self.parseAllIssuesPhase(dictRawIssues, issueTemplate)
        with open(self.finalJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
            jsonOut.writelines(recordCollection.model_dump_json(indent=2))

        self.updateStats(topKey = "Final JSON", keyValList = [ ("Time", time.time() - startTime),  ("Records", recordCollection.objectCount()), ("Input Tokens", totalUsage.input_tokens), ("Output Tokens", totalUsage.output_tokens) ])


    def finalJSONfromRawPhaseAllFiles(self, inputFileList : List[str]):
        """
        Perform conversion of raw JSON to final JSON for all files


        Args:
            inputFileList (List[str]) - list of files
        
        Returns:
        """
        for inputFileName in inputFileList:
            self.updateStats(topKey = "Final JSON", keyValList = [ ("Final JSON", inputFileName)])
            self.inputFileName = self.documentFolder + inputFileName
            self.inputFileBaseName = str(Path(inputFileName).name)
            self.rawTextFromDoc = self.dataFolder + inputFileName + ".raw.txt"
            self.rawJSON = self.dataFolder + inputFileName + ".raw.json"
            self.finalJSON = self.dataFolder + inputFileName + ".json"

            templateInstance = self.openTemplateFile(self.inputFileBaseName)
            self.finalJSONfromRawPhase(issueTemplate = templateInstance) 

            print(self.showStats(topKey = "Final JSON", showKey = "Final JSON", label="Final JSON"))
        self.removeStats(topKey = "Final JSON", removeKey = "Final JSON")


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
            msg = f"prepareBM25corpusPhase: {fileContentOrError} - perform '--finaljson' phase first"
            self.workerSnapshot(msg)
            return corpus
        else:
            recordCollection = RecordCollection.model_validate_json(fileContentOrError)

        for key in recordCollection.finding_dict:
            issue = recordCollection.finding_dict[key]
            reportItem = issueTemplate.model_validate(issue)
            issueText = reportItem.bm25s()
            corpus.append(issueText)

        self.updateStats(topKey = "Prepare BM25 Corpus", keyValList = [ ("Time", time.time() - startTime) ])
        return corpus


    def prepareBM25corpusPhaseAllFiles(self, inputFileList : List[str]) :
        """
        Add issues to BM25s corpus for all files. Create BM25s index.

        Args:
            inputFileList (List[str]) - list of files
        
        Returns:

        """

        # bm25s index is common for all source documents
        corpus : List[str] = []

        for inputFileName in inputFileList:
            self.inputFileName = self.documentFolder + inputFileName
            self.inputFileBaseName = str(Path(inputFileName).name)
            self.rawTextFromDoc = self.dataFolder + inputFileName + ".raw.txt"
            self.rawJSON = self.dataFolder + inputFileName + ".raw.json"
            self.finalJSON = self.dataFolder + inputFileName + ".json"

            templateInstance = self.openTemplateFile(self.inputFileBaseName)
            corpus = self.prepareBM25corpusPhase(issueTemplate = templateInstance, corpus=corpus)

#        stemmer = Stemmer.Stemmer("english")
#        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")
        retriever = bm25s.BM25(corpus=corpus)
        retriever.index(corpus_tokens)
        retriever.save(self.bm25IndexFolder)


    def showConfiguration(self) :
        print(f"Verbosity:\t{CommonHelper.convertLoggingLevel2Name(self.logginglevel)}")
        print(f"Status file:\t{self.statusFileName}")
        print(f"Documents:\t{self.documentFolder}")
        print(f"RAG database:\t{self.ragDatapath}")
        print(f"Interim data:\t{self.dataFolder}")
        print(f"Template file:\t{self.templateJSONName}")
        print(f"BM25s folder:\t{self.bm25IndexFolder}")


    def threadWorker(self, fileList : List[str]):
        """
        Workflow to read, parse, vectorize records
        
        Args:
            fileList (List[str]) - list of files to process
        
        Returns:
            None
        """

        totalStart = time.time()

        if self.loadDocument:
            self.loadDocumentPhaseAllFiles(inputFileList = fileList)
        if self.rawTextFromDocument :
            self.rawTextFromDocumentPhaseAllFiles(inputFileList = fileList)
        if self.finalJSONfromRaw :
            self.finalJSONfromRawPhaseAllFiles(inputFileList = fileList)
        if self.prepareBM25corpus :
            self.prepareBM25corpusPhaseAllFiles(inputFileList = fileList)
        if self.vectorizeFinalJSON :
            self.vectorizeFinalJSONPhaseAllFiles(inputFileList = fileList)
        if self.clear :
            self.clearAllFiles()

        self.updateStats(topKey = "Total", keyValList = [("Time", time.time() - totalStart)])

        pprint(self.stats)

