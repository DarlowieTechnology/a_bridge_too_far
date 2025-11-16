#
# Query workflow class used by Django app and command line
#
import json
from logging import Logger

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

from pydantic import BaseModel
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

from jira import JIRA
from openai import OpenAI


# local
from common import OneResultWithType, ResultWithTypeList
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
            msg = f"Opened report issue collection with {self._chromaReportIssues.count()} documents."
            self.workerSnapshot(msg)
        except Exception as e:
            msg = f"Error: exception opening report issue collection: {e}"
            self.workerError(msg)
            return False
        
        # open jira table
        collectionName = self._config["rag_indexer_jira"]
        try:
            self._chromaJiraItems = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
            msg = f"Opened Jira item collection with {self._chromaJiraItems.count()} documents."
            self.workerSnapshot(msg)
        except Exception as e:
            msg = f"Error: exception opening Jira item collection: {e}"
            self.workerError(msg)
            return False
        return True
    

    def prompt(self, userPrompt : str) -> bool:
        queryString = input(userPrompt)
        if queryString == "c":
            return False
        
        totals = set()
        queryResult = self._chromaReportIssues.query(query_texts=[queryString], n_results=10)
        print(json.dumps(queryResult, indent=2))
        cutDist = 0.7
        resultIdx = -1
        for distFloat in queryResult["distances"][0] :
            resultIdx += 1
            docText = ""
            if (queryResult["documents"]) :
                docText = queryResult["documents"][0][resultIdx]

            if (distFloat > cutDist) :
                break

            totals.add(docText)

        if not len(totals) :
            self._logger.info(f"Query {queryString} did not get matches less than {self._config['rag_distmatch']}")
        else:
            oneResultList = OneResultWithType(results_list=list(totals))
            oneResultList.model_dump_json(indent = 2)

        return True

    def agentPromptOllama(self):

#        queryString = input(userPrompt)
#        if queryString == "c":
#            return False

        query = "XSS or Reflected"
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

        ollModel = OpenAIModel(model_name=self._config["main_llm_name"], 
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
        
    def agentPromptChatGPT(self):
        systemPrompt = f"""
        You are an expert in PCI DSS standard.
        Explain why the vulnerability described in user prompt makes application non-compliant with PCI DSS requirements. 
        List relevant PCI DSS requirements.
        Limit output to one paragraph.
        """

        client = OpenAI()

        # Call the Chat Completion endpoint
        response = client.chat.completions.create(
            model=self._context["llmChatGPTVersion"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how do I connect to ChatGPT with Python?"}
            ]
        )
        # Print the output
        print(response.choices[0].message.content)


    def agentPromptGemini(self):

#        query = input("Enter query or c to cancel:")
#        if query == "c":
#            return False

        oneIssue = None
        oneJiraItem = None
        query = "Reflected Cross-Site Scripting"
        oneResultList = self.getDBIssueMatch(query)
        item = oneResultList.results_list[0]
        IssueTemplate = ParserClassFactory.factory(item.parser_typename)
        oneIssue = IssueTemplate.model_validate_json(item.data)
#        print(f"{oneIssue.model_dump_json(indent=2)}")
        oneJiraResultList = self.getDBJiraMatch(oneIssue)
        jiraItem = oneJiraResultList.results_list[0]
        JiraTemplate = ParserClassFactory.factory(jiraItem.parser_typename)
        oneJiraItem = JiraTemplate.model_validate_json(jiraItem.data)
#        print(f"{oneJiraItem.model_dump_json(indent=2)}")

        print(f"=========RAG Output=====\n\nIdentifier: {oneIssue.identifier}\nTitle:{oneIssue.title}\nRisk:{oneIssue.risk}\nAffects:{oneIssue.affects}")


#        systemPrompt = f"""
#            You are an expert in PCI DSS standard. 
#            Describe PCI DSS Requirement 6.5.7 Cross-Side Scripting compliance requirements.
#            Incorporate in the answer vulnerabilities descriptions from user prompt as an example of non-compliance.
#            Here is the JSON schema for the vulnerability record:
#            {json.dumps(IssueTemplate.model_json_schema(), indent=2)}            
#            """

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

        api_key = self._config["gemini_key"]

        openAIClient = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

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
            print("====Jira Item=====\nXSS is found by external test. Item is recorded in Jira as not fixed\n=======")


        # map Open AI usage to Pydantic usage            
        usage = Usage()
        usage.requests = 1
        usage.request_tokens = completion.usage.prompt_tokens
        usage.response_tokens = completion.usage.completion_tokens
            
        return oneIssue, usage



    def getDBIssueMatch(self, queryString : str) -> ResultWithTypeList :
        """
        Query collection and select vectors within cut-off distance
        
        Args:
            queryString (str) - query

        Returns:
            ResultWithTypeList
        """

        resultWithTypeList = ResultWithTypeList(results_list = [])
        queryResult = self._chromaReportIssues.query(query_texts=[queryString], n_results=1000)
        cutDist = 0.99
        resultIdx = -1
        for distFloat in queryResult["distances"][0] :
            resultIdx += 1
            if (distFloat > cutDist) :
                break
            # result from RAG table has typename attached as metadata
            oneResultWithType = OneResultWithType(
                data = queryResult["documents"][0][0], 
                parser_typename = queryResult["metadatas"][0][0]["recordType"]
            )
            recordHash = hash(oneResultWithType)
            toAdd = True
            for existingItem in resultWithTypeList.results_list:
                if recordHash == hash(existingItem):
                    toAdd = False
                    break
            if toAdd:
                resultWithTypeList.results_list.append(oneResultWithType)

        if not len(resultWithTypeList.results_list) :
            msg = f"Query {queryString} did not get matches less than {cutDist}"
            self.workerError(msg)
            return ResultWithTypeList(results_list = [])
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
        Workflow to perform query. Workflow starts when query is received.
        
        Args:
            None
        
        Returns:
            None

        """

        self._context["stage"] = "startup"
        if not self.startup():
            return

        if self._context["llmProvider"] == "Gemini":
            self.agentPromptGemini()

        self._context["stage"] = "completed"
        msg = f"Processing completed."
        self.workerSnapshot(msg)
