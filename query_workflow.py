#
# Query workflow class used by Django app and command line
#
import json
from logging import Logger
from typing import List
from pathlib import Path
import re


import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic import BaseModel
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

from jira import JIRA
from openai import OpenAI

import Stemmer

import bm25s

import numpy as np
from scipy import stats

import spacy

# local
from common import OneResultWithType, ResultWithTypeList, OneResultList
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory


class QueryWorkflow(WorkflowBase):

    _chromaReportIssues: chromadb.Collection = None
    _chromaJiraItems: chromadb.Collection = None


    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context, logger)


    def startup(self) -> bool:
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
            return False
        
        ef = OllamaEmbeddingFunction(
            model_name=self._config["rag_embed_llm"],
            url=self._config["rag_embed_url"]    
        )

        # open issues table
        collectionName = self._config["rag_indexer_reports"]
        try:
            self._chromaReportIssues = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
            msg = f"Opened report collection with {self._chromaReportIssues.count()} documents."
            self.workerSnapshot(msg)
        except Exception as e:
            msg = f"Error: exception opening report collection: {e}"
            self.workerError(msg)
            return False
        
        # open jira table
        collectionName = self._config["rag_indexer_jira"]
        try:
            self._chromaJiraItems = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
            msg = f"Opened Jira collection with {self._chromaJiraItems.count()} documents."
            self.workerSnapshot(msg)
        except Exception as e:
            msg = f"Error: exception opening Jira collection: {e}"
            self.workerError(msg)
            return False
        
        self._llmModel = None

        if self._context["llmProvider"] == "Ollama":
            self._llmModel = OpenAIModel(model_name=self._context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self._context["llmBaseUrl"]))
        if self._context["llmProvider"] == "Gemini":
            self._llmModel = GeminiModel(
                model_name=self._context["llmGeminiVersion"], 
                provider=GoogleGLAProvider(
                    api_key = self._config["gemini_key"]
                ),
                settings=ModelSettings(temperature = 0.0)
            )

        return True





    def preprocessQuery(self) :
        """
        Preprocess query string with lowercase, remove whitespace, normalise spaces
        
        Args:
            None
        Returns:
            updated query string

        """
        query = self._context["query"]
        query = query.strip().lower()
        query = re.sub(r'[^\w\s?!]', '', query)
        self._context["query"] = " ".join(query.split())
        return self._context["query"]


    def compressQuery(self) :
        """
        Perform Telegraphic Semantic Compression (TSC) on the query
        remove predictable grammar, keep facts.
        ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/
        get english dictionary: python -m spacy download en_core_web_sm
        """

        query = self._context["query"]

        # Load spaCy English model
        nlp = spacy.load("en_core_web_sm")

        # Parts of speech to remove (predictable grammar)
        REMOVE_POS = {"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART"}

        # Optional low-information words to remove
        REMOVE_LIKE = {"like", "just", "really", "basically", "literally"}

        doc = nlp(query)
        chunks = []

        for sent in doc.sents:
            words = [
                token.lemma_
                for token in sent
                if (
                    token.pos_ not in REMOVE_POS
                    and token.text.lower() not in REMOVE_LIKE
                    and not token.is_punct
                )
            ]
            if words:
                chunks.append(" ".join(words))
        self._context["query"] =  " ".join(chunks)

    
    def tokenizeQuery(self, useStopWords : bool = True, useStemmer : bool = False) :

        query = self._context["query"]
        if (useStopWords and useStemmer):
            query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        if (useStopWords and not useStemmer):
            query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en")
        if (not useStopWords and useStemmer):
            query_tokens = bm25s.tokenize(query, return_ids=False, stemmer=Stemmer.Stemmer("english"))
        if (not useStopWords and not useStemmer):
            query_tokens = bm25s.tokenize(query, return_ids=False)
        self._context["query"] = query_tokens


    def hydeQuery(self) :
        """ use HyDE (Hypothetical Document Embedding) """

        systemPrompt = f"Write a two sentence answer to the user prompt query"
#        systemPrompt = f"Generate a hypothetical best-guess answer to user prompt query"

        agentHyDE = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = f"{self._context["query"]}"
        try:
            result = agentHyDE.run_sync(userPrompt)
            self._context["query"] = result.output
            if result.usage():
                self._context["llmrequests"] += 1
                self._context["llmrequesttokens"] += result.usage().request_tokens
                self._context["llmresponsetokens"] += result.usage().response_tokens
        except Exception as e:
            msg = f"LLM exception on HyDE request: {e}"
            self.workerError(msg)


    def multiQuery(self) :
        """use LLM to generate multiple queries form the original query"""

        # Prompt for generating multiple queries
        systemPrompt = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Respond only with a list of questions. Format output as Python list.
        Original query is supplied in user prompt"""

        agentMultipleQ = Agent(self._llmModel, 
                               system_prompt = systemPrompt)
        userPrompt = f"{self._context["query"]}"
        try:
            result = agentMultipleQ.run_sync(userPrompt)
            self._context["query"] = result.output
            if result.usage():
                self._context["llmrequests"] += 1
                self._context["llmrequesttokens"] += result.usage().request_tokens
                self._context["llmresponsetokens"] += result.usage().response_tokens
        except Exception as e:
            msg = f"LLM exception on multi query request: {e}"
            self.workerError(msg)



    def rewriteQuery(self) :
        """Rewrite the query for better retrieval."""

        # Query rewriting prompt
        systemPrompt = """You are a query rewriting expert. The user's original query didn't retrieve relevant documents.

            Analyze the query and rewrite it to improve retrieval chances:
            - Make it more specific or more general as appropriate
            - Add synonyms or related terms
            - Rephrase to target likely document content
            - Consider the retrieval failure and adjust accordingly

            Original query is supplied in user prompt.
            """
        
        agentRewriteQ = Agent(self._llmModel, 
                               system_prompt = systemPrompt)
        userPrompt = f"{self._context["query"]}"
        try:
            result = agentRewriteQ.run_sync(userPrompt)
            self._context["query"] = result.output
            if result.usage():
                self._context["llmrequests"] += 1
                self._context["llmrequesttokens"] += result.usage().request_tokens
                self._context["llmresponsetokens"] += result.usage().response_tokens
        except Exception as e:
            msg = f"LLM exception on rewrite query request: {e}"
            self.workerError(msg)


    def bm25sQuery(self) : 

        query_tokens = self._context["query"]

        # for all data sources for bm25s
        scoresForStats = []
        for folderName in self._context["bm25sJSON"]:
            # first read bm25s configuration for the document
            with open(f"{folderName}/params.index.json", "r", encoding='utf8', errors='ignore') as jsonIn:
                bm25sParams = json.load(jsonIn)
            retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)
            numDocs = bm25sParams["num_docs"]
            results, scores = retriever.retrieve(query_tokens, k=numDocs)
            for i in range(results.shape[1]):
                docN, score = results[0, i], scores[0, i]
                if (score > 0):
                    scoresForStats.append(score)

        a1F = np.array(scoresForStats, dtype=np.float32)
        statLength = len(a1F)
        statMin = np.min(a1F)
        statMax = np.max(a1F)
        statAvg = np.average(a1F)
        statMean = np.mean(a1F)
        statMedian = np.median(a1F)
        statRange = np.max(a1F)-np.min(a1F)
        d2 = abs(a1F - statMean)**2
        statVar = d2.sum() / (statLength)
        statStdDev = statVar**0.5
        statHist5 = np.histogram(a1F, 5)
        statHist10 = np.histogram(a1F, 10)

        msg = f"Length {statLength} Min {statMin}  Max {statMax}  Average {statAvg}  Mean {statMean}  Median {statMedian}"
        self.workerSnapshot(msg)
        msg = f"Range {statRange}  Variance {statVar}  Std Deviation {statStdDev}"
        self.workerSnapshot(msg)
        msg = f"Histogram(5) {statHist5}"
        self.workerSnapshot(msg)
        msg = f"Histogram(10) {statHist10}"
        self.workerSnapshot(msg)


    def vectorQuery(self) -> OneResultList :

        query_tokens = self._context["query"]

        oneResultList = self.getDBIssueMatch(query_tokens)
        return oneResultList




    def agentPromptOllama(self):
        """
        Use Ollama host, embedded vector database, bm25s tokens to retrieve query results
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        
        """
       
        query_tokens = self._context["query"]


        ollModel = OpenAIModel(model_name=self._context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self._context["llmBaseUrl"]))

        oneResultList = self.getDBIssueMatch(query_tokens)
        for item in oneResultList.results_list:
            IssueTemplate = ParserClassFactory.factory(oneResultList.results_list[0].parser_typename)
            oneIssue = IssueTemplate.model_validate_json(item.data)
            print(f"{oneIssue.title}")
        return







        oneResultList = self.getDBIssueMatch(query)
        IssueTemplate = ParserClassFactory.factory(oneResultList.results_list[0].parser_typename)
        oneIssue = IssueTemplate.model_validate_json(oneResultList.results_list[0].data)

        systemPrompt = f"""
        You are an expert in PCI DSS standard.
        Explain why the vulnerability described in user prompt makes application non-compliant with PCI DSS requirements. 
        List relevant PCI DSS requirements.
        Limit output to one paragraph.
        Here is the JSON schema for the vulnerability record:
        {json.dumps(IssueTemplate.model_json_schema(), indent=2)}            
        """
 #       print(f"======System prompt=======\n\n{systemPrompt}")

        userPrompt = f"{oneIssue.model_dump_json(indent=2)}"
        print(f"======User prompt=======\n\n{userPrompt}")

        ollModel = OpenAIModel(model_name=self._context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self._context["llmBaseUrl"]))
        agent = Agent(ollModel,
                    system_prompt = systemPrompt)
        try:
            result = agent.run_sync(userPrompt)
            print(f"=====OUT=====\n\n{result.output}")
            runUsage = result.usage()
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            msg = f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior"
            self.workerError(msg)
            return
        

    def agentPromptGemini(self):
        """
        Compile list of synonyms to query
        Get RAG data for list of synonyms
        Ask Gemini for final reply
        """

        geminiModel = GeminiModel(
            model_name=self._context["llmGeminiVersion"], 
            provider=GoogleGLAProvider(
                api_key = self._config["gemini_key"]
            ),
            settings=ModelSettings(temperature = 0.0)
        )

        # use HyDE (Hypothetical Document Embedding)
        systemPrompt = f"""
        Write a two sentence answer to the user prompt query
        """

        agentHyDE = Agent(geminiModel, system_prompt = systemPrompt)
        userPrompt = f"{self._context["query"]}"
        try:
            result = agentHyDE.run_sync(userPrompt)
        except Exception as e:
            msg = f"Gemini exception on HyDE request: {e}"
            self.workerError(msg)
            return
       
        msg = f"HyDE:{result.output}"
        self.workerSnapshot(msg)

        queryList = []
 #       queryList.append(self._context["query"])
        queryList.append(result.output)
        oneResultList = self.getDBIssueMatch(queryList)
#        print(f"'n---------------\n{oneResultList.model_dump_json(indent=2)}")

        systemPrompt = f"""
        return subcategories and variants of user prompt.
        Here is the JSON schema for the output:
        {json.dumps(OneResultList.model_json_schema(), indent=2)}            
        """

        agentSynonyms = Agent(
            geminiModel,
            output_type=OneResultList,
            system_prompt = systemPrompt)
        userPrompt = f"{self._context["query"]}"
        try:
            result = agentSynonyms.run_sync(userPrompt)
            synonymsResultList = OneResultList.model_validate_json(result.output.model_dump_json())
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerError(msg)
            return

#        synonymsResultList.results_list.append(self._context["query"])
        msg = f"Expanded bm25s query: {json.dumps(synonymsResultList.results_list)}"
        self.workerSnapshot(msg)


#        query_tokens = bm25s.tokenize(synonymsResultList.results_list, stemmer=Stemmer.Stemmer("english"), return_ids=False)
        query_tokens = bm25s.tokenize(synonymsResultList.results_list, return_ids=False)

        msg = f"Tokenised bm25s query: {json.dumps(query_tokens)}"
        self.workerSnapshot(msg)

        # for all data sources for bm25s
        for folderName in self._context["bm25sJSON"]:
            # first read bm25s configuration for the document
            with open(f"{folderName}/params.index.json", "r", encoding='utf8', errors='ignore') as jsonIn:
                bm25sParams = json.load(jsonIn)
            retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)
            numDocs = bm25sParams["num_docs"]
            results, scores = retriever.retrieve(query_tokens, k=numDocs)
            for i in range(results.shape[1]):
                docN, score = results[0, i], scores[0, i]
                if score > self._context["bm25sCutOffScore"]:
                    outTitle = docN["text"].splitlines()[0]
                    folderBaseName = str(Path(folderName).name)
                    msg = f"{folderBaseName}   {outTitle}  Rank: {i+1} Score: {score:.2f})"
                    self.workerSnapshot(msg)

        return

        resultWithTypeList = ResultWithTypeList(results_list = [])
        for query in synList:
            oneResultList = self.getDBIssueMatch(query)
            if len(oneResultList.results_list):
                for item in oneResultList.results_list:
                    recordHash = hash(item)
                    toAdd = True
                    for existingItem in resultWithTypeList.results_list:
                        if recordHash == hash(existingItem):
                            toAdd = False
                            break
                    if toAdd:
                        resultWithTypeList.results_list.append(item)



        msg = f"Found {len(resultWithTypeList.results_list)} issue related to query list"
        self.workerSnapshot(msg)

        IssueTemplate = None
        for item in resultWithTypeList.results_list:
            IssueTemplate = ParserClassFactory.factory(item.parser_typename)
            oneIssue = IssueTemplate.model_validate_json(item.data)
#            print(oneIssue.model_dump_json(indent=2))

        systemPrompt = f"""
        You are an expert in PCI DSS standard.
        Explain why the vulnerability described in user prompt makes 
        application non-compliant with PCI DSS requirements. 
        List relevant PCI DSS requirements.
        Limit output to one paragraph.
        Here is the JSON schema for the vulnerability record:
        {json.dumps(IssueTemplate.model_json_schema(), indent=2)}            
        """

        agent = Agent(
            geminiModel,
            output_type=OneResultList,
            system_prompt = systemPrompt,
            retries=3,
            output_retries=3)

        userPrompt = f"{oneIssue.model_dump_json(indent=2)}"

        try:
            result = agent.run_sync(userPrompt)
            oneIssue = OneResultList.model_validate_json(result.output.model_dump_json())
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            return None, None

        print(f"\n------Gemini Reply---------\n{oneIssue.model_dump_json(indent=2)}")
        return






        oneResultList = self.getDBIssueMatch(query)
        item = oneResultList.results_list[0]
        IssueTemplate = ParserClassFactory.factory(item.parser_typename)
        oneIssue = IssueTemplate.model_validate_json(item.data)

        msg = f"Found issue related to query" 
        self.workerSnapshot(msg)
#        print(f"{oneIssue.model_dump_json(indent=2)}")

        oneJiraResultList = self.getDBJiraMatch(oneIssue)
        jiraItem = oneJiraResultList.results_list[0]
        JiraTemplate = ParserClassFactory.factory(jiraItem.parser_typename)
        oneJiraItem = JiraTemplate.model_validate_json(jiraItem.data)
#        print(f"{oneJiraItem.model_dump_json(indent=2)}")
#        print(f"=========RAG Output=====\n\nIdentifier: {oneIssue.identifier}\nTitle:{oneIssue.title}\nRisk:{oneIssue.risk}\nAffects:{oneIssue.affects}")
        msg = f"Found jira ticket related to query" 
        self.workerSnapshot(msg)

        systemPrompt = f"""
        You are an expert in PCI DSS standard.
        Explain why the vulnerability described in user prompt makes application non-compliant with PCI DSS requirements. 
        List relevant PCI DSS requirements.
        Limit output to one paragraph.
        Here is the JSON schema for the vulnerability record:
        {json.dumps(IssueTemplate.model_json_schema(), indent=2)}            
        """

#        print(f"======System prompt=======\n\n{systemPrompt}")

        userPrompt = f"{oneIssue.model_dump_json(indent=2)}"
 
 #       print(f"======User prompt=======\n{userPrompt}\n==============")

        try:
            completion = openAIClient.beta.chat.completions.parse(
                model = self._context["llmGeminiVersion"],
                temperature=0.0,
                messages=[
                    {"role": "system", "content": f"{systemPrompt}"},
                    {"role": "user", "content": f"{userPrompt}"},
                ]
            )
            geminiResult = completion.choices[0].message.content
            if not geminiResult:
                msg = f"Gemini API error"
                self.workerError(msg)
                return None, None
            
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerSnapshot(msg)
            return None, None

        print(f"======Gemini Output=======\n\n{geminiResult}")

        if oneJiraItem.status_name == "To Do":
            msg = f"Item is recorded in Jira as not fixed"
            self.workerSnapshot(msg)

        # map Open AI usage to Pydantic usage            
        usage = Usage()
        usage.requests = 1
        usage.request_tokens = completion.usage.prompt_tokens
        usage.response_tokens = completion.usage.completion_tokens
            
        return oneIssue, usage



    def getDBIssueMatch(self, queryList : list[str]) -> ResultWithTypeList :
        """
        Query collection and select vectors within cut-off distance
        
        Args:
            queryList (list[str]) - query string list

        Returns:
            ResultWithTypeList
        """

        resultWithTypeList = ResultWithTypeList(results_list = [])
        queryResult = self._chromaReportIssues.query(query_texts=queryList, n_results=1000)
        cutDist = self._context['cutIssueDistance']
        resultIdx = -1

        for distFloat in queryResult["distances"][0]:
            resultIdx += 1
            if (distFloat > cutDist) :
                break

#            print(f"------dist {distFloat}-------------------")
#            print(type(queryResult["documents"][0][resultIdx]))
#            print(queryResult["documents"][0][resultIdx])
#            print("-------------------------")
#            print(type(queryResult["metadatas"][0][resultIdx]))
#            print(queryResult["metadatas"][0][resultIdx])

            # result from RAG table has typename attached as metadata
            oneResultWithType = OneResultWithType(
                data = queryResult["documents"][0][resultIdx], 
                parser_typename = queryResult["metadatas"][0][resultIdx]["recordType"],
                vector_dist = distFloat
            )

            recordHash = hash(oneResultWithType)
            toAdd = True
            for existingItem in resultWithTypeList.results_list:
                if recordHash == hash(existingItem):
                    toAdd = False
                    break
            if toAdd:
                resultWithTypeList.results_list.append(oneResultWithType)

        if len(resultWithTypeList.results_list) :
            msg = f"Query returned {len(resultWithTypeList.results_list)} vectors within the distance of {cutDist}"
            self.workerError(msg)
        else:
            msg = f"Query {queryList} did not return vectors within the distance of {cutDist}"
            self.workerError(msg)
        return resultWithTypeList


    def getDBJiraMatch(self, issueTemplate : BaseModel) -> ResultWithTypeList :
        """
        Query jira item collection and select vectors within cut-off distance
        
        Args:
            issueTemplate - issue from report table

        Returns:
            ResultWithTypeList
        """

        jiraWithTypeList = ResultWithTypeList(results_list = [])

        if hasattr(issueTemplate, "identifier"):
            queryString = issueTemplate.identifier
        else:
            msg = f"Jira Query cannot find identifier field"
            self.workerError(msg)
            return jiraWithTypeList

        queryResult = self._chromaJiraItems.query(query_texts=[queryString], n_results=1000)
        cutDist = 0.99
        resultIdx = -1
        for distFloat in queryResult["distances"][0] :
            resultIdx += 1
            if (distFloat > cutDist) :
                break
            # result from RAG issues table has attached typename for the conversion
            oneResultWithType = OneResultWithType(
                data = queryResult["documents"][0][0], 
                parser_typename = queryResult["metadatas"][0][0]["recordType"]
            )
            recordHash = hash(oneResultWithType)
            toAdd = True
            for existingItem in jiraWithTypeList.results_list:
                if recordHash == hash(existingItem):
                    toAdd = False
                    break
            if toAdd:
                jiraWithTypeList.results_list.append(oneResultWithType)

        if not len(jiraWithTypeList.results_list) :
            msg = f"Jira query {queryString} did not get matches less than {cutDist}"
            self.workerError(msg)
        return jiraWithTypeList


    def threadWorker(self):
        """
        Workflow to perform query. 
        
        Args:
            None
        
        Returns:
            None

        """

        if not self.startup():
            msg = f"ERROR: Cannot initialize RAG database"
            self.workerError(msg)
            return

        if self._context["llmProvider"] == "Gemini":
            self.agentPromptGemini()
        if self._context["llmProvider"] == "Ollama":
            self.agentPromptOllama()

        self._context["stage"] = "completed"
        msg = f"Processing completed."
        self.workerSnapshot(msg)
