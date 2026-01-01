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
from common import COLLECTION, OneResultWithType, ResultWithTypeList, OneResultList, OneQueryBM25SAppResult, OneQuerySemanticAppResult, AllQueryAppResults, StatsOnResults
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory


class QueryWorkflow(WorkflowBase):

    def __init__(self, context : dict, logger : Logger):
        """
        Args:
            context (dict)
            logger (Logger) - can originate in CLI or Django app
        """
        super().__init__(context=context, logger=logger, createCollection=False)


    def startup(self) -> bool:
        """
        create LLM access object on startup
        
        :return: True if LLM object is created
        :rtype: bool
        """        
        self._llmModel = None

        if self.context["llmProvider"] == "Ollama":
            self._llmModel = OpenAIModel(model_name=self.context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self.context["llmBaseUrl"]))
        if self.context["llmProvider"] == "Gemini":
            self._llmModel = GeminiModel(
                model_name=self.context["llmGeminiVersion"], 
                provider=GoogleGLAProvider(
                    api_key = self.config["gemini_key"]
                ),
                settings=ModelSettings(temperature = 0.0)
            )
        return True


    def getQuery(self) -> str :
        return self.context["query"]


    def preprocessQuery(self, queryStr : str) :
        """
        Preprocess query string for search: lowercase, remove whitespace, normalise spaces
        """
        query = queryStr.strip().lower()
        query = re.sub(r'[^\w\s?!]', '', query)
        query = " ".join(query.split())
        return query


    def compressQuery(self, query : str) -> str:
        """
        Perform Telegraphic Semantic Compression (TSC) on the query for semantic search. 
        ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/
        get english dictionary: python -m spacy download en_core_web_sm
        Overwrite self.context['queryCompressed'] with new value.
        """

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
        self.context["queryCompressed"] =  " ".join(chunks)
        return self.context["queryCompressed"]

    
    def tokenizeQuery(self, query : str, useStopWordsFlag : bool = True, useStemmerFlag : bool = False) -> str:
        """
        create list of tokens from the query for BM25S search.
        Overwrite self.context['queryTokenized'] with new value.
        
        :param useStopWordsFlag:  remove stopwords from the query.
        :type useStopWordsFlag: bool
        :param useStemmerFlag: reduce words in query to stems .
        :type useStemmerFlag: bool
        """

        if (useStopWordsFlag and useStemmerFlag):
            query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en", stemmer=Stemmer.Stemmer("english"))
        if (useStopWordsFlag and not useStemmerFlag):
            query_tokens = bm25s.tokenize(query, return_ids=False, stopwords="en")
        if (not useStopWordsFlag and useStemmerFlag):
            query_tokens = bm25s.tokenize(query, return_ids=False, stemmer=Stemmer.Stemmer("english"))
        if (not useStopWordsFlag and not useStemmerFlag):
            query_tokens = bm25s.tokenize(query, return_ids=False)
        self.context["queryTokenized"] = query_tokens
        return query_tokens


    def hydeQuery(self, query : str) -> str:
        """ Use HyDE (Hypothetical Document Embedding) to improve the query for semantic search. 
            Overwrite self.context['queryHyDE'] with new value.
        """

        systemPrompt = f"Write a two sentence answer to the user prompt query"
#        systemPrompt = f"Generate a hypothetical best-guess answer to user prompt query"

        agentHyDE = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentHyDE.run_sync(userPrompt)
            self.context["queryHyDE"] = result.output
            if result.usage():
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += result.usage().request_tokens
                self.context["llmresponsetokens"] += result.usage().response_tokens
            return self.context["queryHyDE"]
        except Exception as e:
            msg = f"LLM exception on HyDE request: {e}"
            self.workerError(msg)
        return ""


    def multiQuery(self, query : str) -> str:
        """Generate multiple queries form the original query for semantic search. 
            Overwrite self.context['queryMultiple'] with new value.
        """

        # Prompt for generating multiple queries
        systemPrompt = """You are an AI language model assistant. Your task is to generate five 
        different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Respond only with a list of questions. Format output as Python list.
        Original query is supplied in user prompt"""

        agentMultipleQ = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentMultipleQ.run_sync(userPrompt)
            self.context["queryMultiple"] = result.output
            if result.usage():
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += result.usage().request_tokens
                self.context["llmresponsetokens"] += result.usage().response_tokens
            return self.context["queryMultiple"]
        except Exception as e:
            msg = f"LLM exception on multi query request: {e}"
            self.workerError(msg)


    def rewriteQuery(self, query : str) -> str:
        """
        Rewrite the query for semantic search. 
        Overwrite context['queryRewrite'] with new value.
        """
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
        userPrompt = query
        try:
            result = agentRewriteQ.run_sync(userPrompt)
            self.context["queryRewrite"] = result.output
            if result.usage():
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += result.usage().request_tokens
                self.context["llmresponsetokens"] += result.usage().response_tokens
            return self.context["queryRewrite"]
        except Exception as e:
            msg = f"LLM exception on rewrite query request: {e}"
            self.workerError(msg)
        return ""


    def prepBM25S(self, query : str) -> str:
        """Prepare query for BM25S search using LLM. 
        """

        # Prompt for generating prepared query
        systemPrompt = """You are a cybersecurity expert. 
        Return a list of terms in the original user prompt with their descriptions.
        Return only the results. Format output as a list of Python strings.
        Original query is supplied in user prompt"""

        agentMultipleQ = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentMultipleQ.run_sync(userPrompt)
            if result.usage():
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += result.usage().request_tokens
                self.context["llmresponsetokens"] += result.usage().response_tokens
            return result.output
        except Exception as e:
            msg = f"LLM exception on prepBM25S query request: {e}"
            self.workerError(msg)



    def statsOnList(self, scoresForStats) -> StatsOnResults:
        """
        Calculate stats on the array of scores
        
        :param scoresForStats: float list
        :return: dataclass with stats
        :rtype: StatsOnResults
        """

        statsOnResults = StatsOnResults()

        if not len(scoresForStats):
            msg = f"Zero length results"
            self.workerSnapshot(msg)
            return statsOnResults

        a1F = np.array(scoresForStats, dtype=np.float32)

        statsOnResults.length = len(a1F)
        statsOnResults.min = np.min(a1F)
        statsOnResults.max = np.max(a1F)
        statsOnResults.avg = np.average(a1F)
        statsOnResults.mean = np.mean(a1F)
        statsOnResults.median = np.median(a1F)
        statsOnResults.range = np.max(a1F)-np.min(a1F)
        statsOnResults.q1, statsOnResults.q2, statsOnResults.q3 = np.quantile(a1F, [0.25, 0.5, 0.75])

#        msg = f"Length {statLength} Min {statMin}  Max {statMax}  Average {statAvg}  Mean {statMean}  Median {statMedian}"
#        self.workerSnapshot(msg)
#        msg = f"Range {statRange}  Variance {statVar}  Std Deviation {statStdDev}"
#        self.workerSnapshot(msg)
#        msg = f"Histogram(5) {statHist5}"
#        self.workerSnapshot(msg)
#        msg = f"Q1: {q1}, Median (Q2): {q2}, Q3: {q3}"
#        self.workerSnapshot(msg)

        return statsOnResults


    def bm25sQuery(self, query : str, allQueryAppResults : AllQueryAppResults, folderName : str) -> AllQueryAppResults : 
        """
        Perform bm25s query for combined corpus of documents saved in the folder
        data in corpus is encoded as 'identifier\ntitle'
        """
        query_tokens = query

        print(f"================Accessing bm25s index in folder: {folderName}")

        # read bm25s configuration for the document
        with open(f"{folderName}/params.index.json", "r", encoding='utf8', errors='ignore') as jsonIn:
            bm25sParams = json.load(jsonIn)
        retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)

        # number of issues in the document
        numDocs = bm25sParams["num_docs"]
        results, scores = retriever.retrieve(query_tokens, k=numDocs)
        for i in range(results.shape[1]):
            docN, score = results[0, i], scores[0, i]
            docN = docN["text"].splitlines()
            if (score > 0):
                oneQueryBM25SAppResult = OneQueryBM25SAppResult(
                    identifier = docN[0].strip(),
                    title = docN[1].strip(),
                    report = str(Path(folderName).stem),
                    score = score
                )
                allQueryAppResults.appendResult(oneQueryBM25SAppResult)
            else:
                print(f"{docN}")

        return allQueryAppResults
    

    def vectorQuery(self, query : str, allQueryAppResults : AllQueryAppResults) -> AllQueryAppResults:
        """
        Perform semantic query 
        """
        cutDist = self.context['cutIssueDistance']
        collectionReports = self.collections[COLLECTION.ISSUES.value]
        queryResult = collectionReports.query(query_texts=query, n_results=1000)

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

            IssueTemplate = ParserClassFactory.factory(queryResult["metadatas"][0][resultIdx]["recordType"])
            oneIssue = IssueTemplate.model_validate_json(queryResult["documents"][0][resultIdx])

            oneQuerySemanticAppResult = OneQuerySemanticAppResult(
                identifier = oneIssue.identifier,
                title = oneIssue.title,
                report = queryResult["metadatas"][0][resultIdx]["document"],
                distanceSemantic = distFloat
            )

            allQueryAppResults.appendResult(oneQuerySemanticAppResult)

        return allQueryAppResults


    def agentPromptOllama(self):
        """
        Use Ollama host, embedded vector database, bm25s tokens to retrieve query results
        
        Args:
            docs (str) - text with unstructured data
            ClassTemplate (BaseModel) - description of structured data

        Returns:
            Tuple of BaseModel and Usage
        
        """
       
        query_tokens = self.context["query"]


        ollModel = OpenAIModel(model_name=self.context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self.context["llmBaseUrl"]))

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

        ollModel = OpenAIModel(model_name=self.context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self.context["llmBaseUrl"]))
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
            model_name=self.context["llmGeminiVersion"], 
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
        userPrompt = f"{self.context["query"]}"
        try:
            result = agentHyDE.run_sync(userPrompt)
        except Exception as e:
            msg = f"Gemini exception on HyDE request: {e}"
            self.workerError(msg)
            return
       
        msg = f"HyDE:{result.output}"
        self.workerSnapshot(msg)

        queryList = []
 #       queryList.append(self.context["query"])
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
        userPrompt = f"{self.context["query"]}"
        try:
            result = agentSynonyms.run_sync(userPrompt)
            synonymsResultList = OneResultList.model_validate_json(result.output.model_dump_json())
        except Exception as e:
            msg = f"Exception: {e}"
            self.workerError(msg)
            return

#        synonymsResultList.results_list.append(self.context["query"])
        msg = f"Expanded bm25s query: {json.dumps(synonymsResultList.results_list)}"
        self.workerSnapshot(msg)


#        query_tokens = bm25s.tokenize(synonymsResultList.results_list, stemmer=Stemmer.Stemmer("english"), return_ids=False)
        query_tokens = bm25s.tokenize(synonymsResultList.results_list, return_ids=False)

        msg = f"Tokenised bm25s query: {json.dumps(query_tokens)}"
        self.workerSnapshot(msg)

        # for all data sources for bm25s
        for folderName in self.context["bm25sJSON"]:
            # first read bm25s configuration for the document
            with open(f"{folderName}/params.index.json", "r", encoding='utf8', errors='ignore') as jsonIn:
                bm25sParams = json.load(jsonIn)
            retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)
            numDocs = bm25sParams["num_docs"]
            results, scores = retriever.retrieve(query_tokens, k=numDocs)
            for i in range(results.shape[1]):
                docN, score = results[0, i], scores[0, i]
                if score > self.context["bm25sCutOffScore"]:
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
                model = self.context["llmGeminiVersion"],
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

        cutDist = self.context['cutIssueDistance']
        queryResult = self._chromaReportIssues.query(query_texts=queryList, n_results=1000)

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

        if self.context["llmProvider"] == "Gemini":
            self.agentPromptGemini()
        if self.context["llmProvider"] == "Ollama":
            self.agentPromptOllama()

        self.context["stage"] = "completed"
        msg = f"Processing completed."
        self.workerSnapshot(msg)
