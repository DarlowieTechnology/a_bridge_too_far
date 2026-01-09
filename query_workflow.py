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
import spacy
from anyascii import anyascii


# local
from common import COLLECTION, QUERYTYPES, TOKENIZERTYPES, OneResultWithType, ResultWithTypeList, OneResultList
from resultsQueryClasses import SEARCH, OneQueryAppResult, OneQueryResultList, AllQueryResults
from workflowbase import WorkflowBase 
from parserClasses import ParserClassFactory

from testQueries import TestQuery, TestSetCollection


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
        Open LLM connection
        
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

    def getQueryTransform(self) -> QUERYTYPES :
        return self.context["querytransforms"]
    
    def getBM25SFolder(self) -> str :
        return self.context["bm25sIndexFolder"]

    def getRRFTopResults(self) -> int :
        return self.context["rrfTopResults"]

    def getTokenizerTypes(self) -> TOKENIZERTYPES:
        return self.context["querybm25options"]


    def preprocessQuery(self, queryStr : str) :
        """
        Preprocess string for interaction with LLM: convert all characters to ASCII, lowercase, remove whitespace, normalise spaces
        Can be used on user input or output of LLM

        :param query: original query 
        :type query: str
        :return: preprocessed query
        :rtype: str
        """
        query = anyascii(queryStr)
        query = query.strip().lower()
        query = re.sub(r'[^\w\s?!]', '', query)
        query = " ".join(query.split())
        return query


    def compressQuery(self, query : str) -> str:
        """
        Perform Telegraphic Semantic Compression (TSC) on the query for semantic search. 
        Ref: https://developer-service.blog/telegraphic-semantic-compression-tsc-a-semantic-compression-method-for-llm-contexts/.
        Get english dictionary: python -m spacy download en_core_web_sm.

        :param query:  original query 
        :type query: str
        :return: compressed query
        :rtype: str
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
        newQuery =  " ".join(chunks)
        return newQuery

    
    def tokenizeQuery(self, query : str, tokenizerTypes: TOKENIZERTYPES) -> str:
        """
        create list of tokens from the query for BM25S search.
        Overwrite self.context['queryTokenized'] with new value.
        
        :param useStopWordsFlag:  remove stopwords from the query.
        :type useStopWordsFlag: bool
        :param useStemmerFlag: reduce words in query to stems .
        :type useStemmerFlag: bool
        """

        if TOKENIZERTYPES.STOPWORDSEN in tokenizerTypes:
            stopwords = "english"
        else:
            stopwords = None

        if TOKENIZERTYPES.STEMMER in tokenizerTypes:
            stemmer=Stemmer.Stemmer("english")
        else:
            stemmer = None

        query_tokens = bm25s.tokenize(query, return_ids=False, stopwords=stopwords, stemmer=stemmer)
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
        Overwrite context['querybm25sprep'] with new value.
        """

        # Prompt for generating prepared query
        systemPrompt = """You are a cybersecurity expert. 
        Return a list of terms in the original user prompt with their descriptions.
        Return only the results. Format output as a list of Python strings.
        Original query is supplied in user prompt"""

        agentPrepBM25s = Agent(self._llmModel, system_prompt = systemPrompt)
        userPrompt = query
        try:
            result = agentPrepBM25s.run_sync(userPrompt)
            self.context['querybm25sprep'] = result.output
            if result.usage():
                self.context["llmrequests"] += 1
                self.context["llmrequesttokens"] += result.usage().request_tokens
                self.context["llmresponsetokens"] += result.usage().response_tokens
            return self.context['querybm25sprep']
        except Exception as e:
            msg = f"LLM exception on prepBM25S query request: {e}"
            self.workerError(msg)



    def bm25sQuery(self, query : str, folderName : str, queryLabel : str) -> OneQueryResultList : 
        """
        Perform bm25s query for combined corpus of documents
        data in corpus is encoded as 'identifier\\ntitle'
        Number of items retrieved is limited to min of context['bm25sRetrieveNum'] and number of items
        Discard items with score less or equal to value context['bm25sCutOffScore']

        :param query: query for bm25s
        :type query: str
        :param folderName: name of folder with bm25s index
        :type folderName: str
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :return: search result object
        :rtype: OneQueryResultList
        """

        oneQueryResultList = OneQueryResultList(
            result_dict = {},
            query = query,
            searchType = SEARCH.BM25S.value,
            label = queryLabel            
        )

        retriever = bm25s.BM25.load(f"{folderName}", mmap=True, load_corpus=True)

        max_items = self.context['bm25sRetrieveNum']
        if retriever.scores["num_docs"] < self.context['bm25sRetrieveNum']:
            max_items = retriever.scores["num_docs"]

        results, scores = retriever.retrieve(query, k=max_items)
        for rankIdx in range(results.shape[1]):
            docN, score = results[0, rankIdx], scores[0, rankIdx]
            docN = docN["text"].splitlines()
            if (score > self.context['bm25sCutOffScore']):
                oneQueryResultList.appendResult(
                    identifier = docN[0].strip(),
                    title = docN[1].strip(),
                    report = str(Path(folderName).stem),
                    score = score,
                    rank = rankIdx + 1
                )
        return oneQueryResultList


    def vectorQuery(self, query : str, collection : COLLECTION, queryLabel : str) -> OneQueryResultList:
        """
        Performs vector (semantic) query. Returns list of results
        Uses context['cutIssueDistance'] to cut results off
        Uses context['semanticRetrieveNum'] to limit max number of items
        
        :param query: query for semantic search
        :type query: str
        :param collection: chroma DB collection name for query
        :type query: COLLECTION
        :param queryLabel: unique label to query run
        :type queryLabel: str
        :return: list of results
        :rtype: OneQueryResultList
        """

        oneQueryResultList = OneQueryResultList(
            result_dict = {},
            query = query,
            searchType = SEARCH.SEMANTIC.value,
            label = queryLabel
        )

        cutDist = self.context['cutIssueDistance']
        chromaCollection = self.collections[collection]
        queryResult = chromaCollection.query(query_texts=query, n_results=self.context['semanticRetrieveNum'])

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

            oneQueryResultList.appendResult(
                identifier = oneIssue.identifier,
                title = oneIssue.title,
                report = queryResult["metadatas"][0][resultIdx]["document"],
                score = distFloat,
                rank = resultIdx + 1
            )
        return oneQueryResultList


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


    def rrfReRanking(self, allQueryResults : AllQueryResults) -> AllQueryResults:
        """
        Reciprocal Rank Fusion (RRF) re-ranking of semantic and bm25s search results
        
        :param allQueryResults: query results
        :type allQueryResults: AllQueryResults
        :return: query results updated with rank
        :rtype: AllQueryResults
        """

    #    for item in allQueryResults.result_lists:
    #        msg = f"RRF:  {item.label} matches: {len(item.result_dict)}"    
    #        queryWorkflow.workerSnapshot(msg)

        # merge keys from all runs into set
        setKeys = set()
        for item in allQueryResults.result_lists:
            for key in item.result_dict:
                setKeys.add(key)

    #    msg = f"RRF: Length of combined keys: {len(setKeys)}"
    #    queryWorkflow.workerSnapshot(msg)
        
        # calculate rank for issue access all query runs
        rrfScores = {}
        for ident in list(setKeys):
            finalRank = 0
            oneQueryAppResult = None
            for item in allQueryResults.result_lists:
                if ident in item.result_dict:
                    oneQueryAppResult = item.result_dict[ident]
                    finalRank += 1/(60 + oneQueryAppResult.rank)
            rrfScores[ident] = [finalRank, oneQueryAppResult]
        # sort descending by rank
        rrfScores = dict(sorted(rrfScores.items(), key=lambda item: item[1][0], reverse=True))
        allQueryResults.rrfScores = rrfScores
        return allQueryResults


    def performQueries(self) -> AllQueryResults :

        allQueryResults = AllQueryResults(
            result_lists = [],
            rrfScores = {}
        )

        originalQuery = self.getQuery()
        queryTransform = self.getQueryTransform()
        bm25sFolder = self.getBM25SFolder()

        msg = f"original: {originalQuery}"
        self.workerSnapshot(msg)

        if QUERYTYPES.ORIGINAL in queryTransform:
            if self.context['queryPreprocess']:
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed: {originalQuery}"
                self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(originalQuery, COLLECTION.ISSUES.value, "ORIG"))

        if QUERYTYPES.ORIGINALCOMPRESS in queryTransform:
            if self.context['queryPreprocess']:
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed: {originalQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(originalQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "ORIGCOMPRESS"))

        if QUERYTYPES.HYDE in queryTransform:
            hydeQuery = self.hydeQuery(originalQuery)
            msg = f"hyde: {hydeQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                hydeQuery = self.preprocessQuery(hydeQuery)
                msg = f"preprocessed: {hydeQuery}"
                self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(hydeQuery, COLLECTION.ISSUES.value, "HYDE"))

        if QUERYTYPES.HYDECOMPRESS in queryTransform:
            hydeQuery = self.hydeQuery(originalQuery)
            msg = f"hyde: {hydeQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                hydeQuery = self.preprocessQuery(hydeQuery)
                msg = f"preprocessed: {hydeQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(hydeQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "HYDECOMPRESS"))

        if QUERYTYPES.MULTI in queryTransform:
            multiQuery = self.multiQuery(originalQuery)
            msg = f"multi: {json.dumps(multiQuery)}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                multiQuery = self.preprocessQuery(multiQuery)
                msg = f"preprocessed: {multiQuery}"
                self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(multiQuery, COLLECTION.ISSUES.value, "MULTI"))

        if QUERYTYPES.MULTICOMPRESS in queryTransform:
            multiQuery = self.multiQuery(originalQuery)
            msg = f"multi: {json.dumps(multiQuery)}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                multiQuery = self.preprocessQuery(multiQuery)
                msg = f"preprocessed: {multiQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(multiQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(compressedQuery, COLLECTION.ISSUES.value, "MULTICOMPRESS"))

        if QUERYTYPES.REWRITE in queryTransform:
            rewriteQuery = self.rewriteQuery(originalQuery)
            msg = f"rewrite: {rewriteQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                rewriteQuery = self.preprocessQuery(rewriteQuery)
                msg = f"preprocessed: {rewriteQuery}"
                self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(rewriteQuery, COLLECTION.ISSUES.value, "REWRITE"))

        if QUERYTYPES.REWRITECOMPRESS in queryTransform:
            rewriteQuery = self.rewriteQuery(originalQuery)
            msg = f"rewrite: {rewriteQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                rewriteQuery = self.preprocessQuery(rewriteQuery)
                msg = f"preprocessed: {rewriteQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(rewriteQuery)
            msg = f"compress: {compressedQuery}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.vectorQuery(rewriteQuery, COLLECTION.ISSUES.value, "REWRITECOMPRESS"))

        if QUERYTYPES.BM25SORIG in queryTransform:
            if self.context['queryPreprocess']:
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed for BM25s: {originalQuery}"
                self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(originalQuery, self.getTokenizerTypes())
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.bm25sQuery(tokenizedQuery, bm25sFolder, "BM25SORIG"))

        if QUERYTYPES.BM25SORIGCOMPRESS in queryTransform:
            if self.context['queryPreprocess']:
                originalQuery = self.preprocessQuery(originalQuery)
                msg = f"preprocessed for TSC: {originalQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(originalQuery)
            msg = f"compressed for BM25s: {compressedQuery}"
            self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(compressedQuery, self.getTokenizerTypes())
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.bm25sQuery(tokenizedQuery, bm25sFolder, "BM25SORIGCOMPRESS"))

        if QUERYTYPES.BM25PREP in queryTransform:
            bm25sQuery = self.prepBM25S(originalQuery)
    #        bm25sQuery = "['XSS', 'Cross-Site Scripting: A type of web application security vulnerability that allows an attacker to inject malicious code into a vulnerable website, which can then be executed by the user browser.']"
            msg = f"prepared for BM25s: {bm25sQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                bm25sQuery = self.preprocessQuery(bm25sQuery)
                msg = f"preprocessed for BM25s: {bm25sQuery}"
                self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(bm25sQuery, self.getTokenizerTypes())
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.bm25sQuery(tokenizedQuery, bm25sFolder, "BM25SPREP"))

        if QUERYTYPES.BM25PREPCOMPRESS in queryTransform:
            bm25sQuery = self.prepBM25S(originalQuery)
    #        bm25sQuery = "['XSS', 'Cross-Site Scripting: A type of web application security vulnerability that allows an attacker to inject malicious code into a vulnerable website, which can then be executed by the user browser.']"
            msg = f"prepared for TSC: {bm25sQuery}"
            self.workerSnapshot(msg)
            if self.context['queryPreprocess']:
                bm25sQuery = self.preprocessQuery(bm25sQuery)
                msg = f"preprocessed for TSC: {bm25sQuery}"
                self.workerSnapshot(msg)
            compressedQuery = self.compressQuery(bm25sQuery)
            msg = f"compressed for BM25s: {compressedQuery}"
            self.workerSnapshot(msg)
            tokenizedQuery = self.tokenizeQuery(compressedQuery, self.getTokenizerTypes())
            msg = f"tokenized: {json.dumps(tokenizedQuery)}"
            self.workerSnapshot(msg)
            allQueryResults.result_lists.append(self.bm25sQuery(tokenizedQuery, bm25sFolder, "BM25SPREPCOMPRESS"))

        allQueryResults = self.rrfReRanking(allQueryResults)
        return allQueryResults


    def threadWorker(self):
        """
        Workflow to perform query. 
        
        Args:
            None
        
        Returns:
            None

        """

        if not self.startup():
            msg = f"workflow startup failed."
            self.workerError(msg)
            return

        allQueryResults = self.performQueries()

        testQuery = TestSetCollection().getCurrentTest()
        for item in allQueryResults.result_lists:
            msg = testQuery.outputRunInfo(item, item.label)
            self.workerSnapshot(msg)

        msgList = testQuery.outputRRFInfo(allQueryResults.rrfScores, self.getRRFTopResults())
        self.workerResult(msgList)

        score = testQuery.calculateOverallScore(allQueryResults, self.getRRFTopResults()) * 100
        msg = f"Overall score: <b>{score:.4f} %</b>"
        msgList = []
        msgList.append(" ")
        msgList.append(msg)
        self.workerResult(msgList)


#        if self.context["llmProvider"] == "Gemini":
#            self.agentPromptGemini()
#        if self.context["llmProvider"] == "Ollama":
#            self.agentPromptOllama()

        self.context["stage"] = "completed"
        msg = f"Processing completed."
        self.workerSnapshot(msg)

