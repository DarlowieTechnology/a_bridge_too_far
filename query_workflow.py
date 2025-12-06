#
# Query workflow class used by Django app and command line
#
import json
from logging import Logger
from typing import List


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

        # TODO - ask LLM for synonyms
        self._dictSynonyms = {
            "XSS" : [
                "Cross-Site Scripting Attack",
                "cross-site scripting",
                "Cross-Site Scripting (XSS)",
                "HTML Injection",
                "Client-Side Code Injection",
                "JavaScript Injection",
                "DOM-Based Injection",
                "Reflected XSS",
                "Stored XSS",
                "DOM-Based XSS"
            ]
        }


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
        return True
    

    def agentPromptOllama(self):

        query = self._context["query"]
        msg = f"Query: {query}"
        self.workerSnapshot(msg)

        systemPrompt = f"""
        You are an expert in cyber security. Retrieve list of synonyms for a definition. Output just the result.
        Here is the JSON schema for the OneResultList model you must use as context for what information is expected:
        {json.dumps(OneResultList.model_json_schema(), indent=2)}
        """

        userPrompt = f"{query}"

        ollModel = OpenAIModel(model_name=self._context["llmOllamaVersion"], 
                            provider=OpenAIProvider(base_url=self._config["llm_base_url"]))
        agent = Agent(ollModel, 
                    output_type=OneResultList,
                    system_prompt = systemPrompt,
                    retries=3,
                    output_retries=3)

        try:
            result = agent.run_sync(userPrompt)
            oneResultList = OneResultList.model_validate_json(result.output.model_dump_json())
        except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
            msg = f"extractExecSection: Skipping due to exception: {e}"
            self.workerError(msg)
            return None, None

        self._logger.info(f"\n------{OneResultList}---------\n{oneResultList.model_dump_json(indent=2)}")
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
                            provider=OpenAIProvider(base_url=self._config["llm_base_url"]))
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

        msg = f"Original query:{self._context["query"]}"
        self.workerSnapshot(msg)

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
       
        msg = f"HyDE response:{result.output}"
        self.workerSnapshot(msg)

        queryList = []
        queryList.append(self._context["query"])
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

        synonymsResultList.results_list.append(self._context["query"])
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
                outTitle = docN["text"].splitlines()[0]
                if score > self._context["bm25sCutOffScore"]:
                    msg = f"{folderName}  Rank {i+1} (score: {score:.2f}): {outTitle}"
                    self.workerSnapshot(msg)

        return




        synList = []
        synList.append(self._context["query"])
        synList = synList + self._dictSynonyms[self._context["query"]]
        msg = f"Query: {synList}"
        self.workerSnapshot(msg)

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

#            print("-------------------------")
#            print(type(queryResult["documents"][0][resultIdx]))
#            print(queryResult["documents"][0][resultIdx])
#            print("-------------------------")
#            print(type(queryResult["metadatas"][0][resultIdx]))
#            print(queryResult["metadatas"][0][resultIdx])
#            print("============================")

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
            msg = f"Query returned {len(resultWithTypeList.results_list)} vectors less than {cutDist} distance"
            self.workerError(msg)
        else:
            msg = f"Query {queryList} did not get matches less than {cutDist}"
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

        self._context["stage"] = "completed"
        msg = f"Processing completed."
        self.workerSnapshot(msg)
