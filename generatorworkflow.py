#
# generator workflow class used by Django app and command line
#
import sys
import logging
from logging import Logger
import json
import tomli
import re
import time


from pydantic import BaseModel, Field
import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage


# local
sys.path.append("..")
sys.path.append("../..")

# local
from common import ConfigSingleton, DebugUtils, ReportIssue, AllReportIssues, OpenFile


class GeneratorWorkflow(BaseModel):

    def workerSnapshot(self, logger : Logger, fileName : str, context : dict, msg : str):
        """
        Logs status and updates status file

        Args:
            logger (Logger) - logger object
            fileName (str) - status file name
            context (dict) - process context data
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            logger.info(msg)
            context['status'].append(msg)
        with open(fileName, "w") as jsonOut:
            formattedOut = json.dumps(context, indent=2)
            jsonOut.write(formattedOut)

    def workerError(self, logger : Logger, fileName : str, context : dict, msg : str):
        """
        Logs error and sets process status to error

        Args:
            logger (Logger) - logger object
            fileName (str) - status file name
            context (dict) - process context data
            msg (str) - message string 

        Returns:
            None
        """
        if msg:
            logger.info(msg)
            context['status'].append(msg)
        context['stage'] = 'error'
        with open(fileName, "w") as jsonOut:
            formattedOut = json.dumps(context, indent=2)
            jsonOut.write(formattedOut)

    def threadWorker(self, sessionKey, fileName, adText):

        logger = logging.getLogger(sessionKey)

        context = {}
        context['status'] = list()
        context["llmrequests"] = 0
        context["llmrequesttokens"] = 0
        context["llmresponsetokens"] = 0

        time.sleep(1)

        #-----------------stage configure

        start = time.time()
        totalStart = start

        context["stage"] = "configure"
        self.workerSnapshot(logger, fileName, context, None)
        
        configName = str(Path(__file__).parent.resolve()) + '/../../default.toml'
        try:
            with open(configName, mode="rb") as fp:
                ConfigSingleton().conf = tomli.load(fp)
        except Exception as e:
            msg = f"Error: config file {configName}, exception {e}"
            self.workerError(logger, fileName, context, msg)
            return

        try:
            chromaClient = chromadb.PersistentClient(
                path=ConfigSingleton().getAbsPath("rag_datapath"),
                settings=Settings(anonymized_telemetry=False),
                tenant=DEFAULT_TENANT,
                database=DEFAULT_DATABASE,
            )
        except Exception as e:
            msg = f"Error: OpenAI API exception: {e}"
            self.workerError(logger, fileName, context, msg)
            return
        
        ef = OllamaEmbeddingFunction(
            model_name=ConfigSingleton().conf["rag_embed_llm"],
            url=ConfigSingleton().conf["rag_embed_url"]    
        )

        collectionName = "actreal"
        try:
            chromaActivity = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
        except Exception as e:
            msg = f"Error: collection ACTIVITY exception: {e}"
            self.workerError(logger, fileName, context, msg)
            return

        collectionName = "scenario"
        try:
            chromaScenario = chromaClient.get_collection(
                name=collectionName,
                embedding_function=ef
            )
        except Exception as e:
            msg = f"Error: collection SCENARIO exception: {e}"
            self.workerError(logger, fileName, context, msg)
            return

        end = time.time()
        msg = f"Opened vector collections ACTIVITY with {chromaActivity.count()} documents, SCENARIO with {chromaScenario.count()} documents. {(end-start):9.4f} seconds"
        self.workerSnapshot(logger, fileName, context, msg)

        #----------------stage summary

        start = time.time()

        context["stage"] = "summary"
        self.workerSnapshot(logger, fileName, context, None)


        jobAdRecord = OneRecord(
            id = "", 
            name=str(sessionKey), 
            description=adText
        )

        execSummary, usageStats = extractExecSection(jobAdRecord, logger, fileName, context)
    #    execSummary, usageStats = fakeExtractExecSection()
        if not execSummary:
            return

        context['jobtitle'] = execSummary.title
        context['execsummary'] = execSummary.description
        if usageStats:
            context["llmrequests"] = usageStats.requests
            context["llmrequesttokens"] = usageStats.request_tokens
            context["llmresponsetokens"] = usageStats.response_tokens

        end = time.time()

        if usageStats:
            msg = f"Extracted executive summary from job description. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
        else:
            msg = f"Extracted executive summary from job description. {(end-start):9.4f} seconds."
        self.workerSnapshot(logger, fileName, context, msg)

        #----------------stage extract

        start = time.time()

        context["stage"] = "extract"
        self.workerSnapshot(logger, fileName, context, None)

        allDescriptions = AllDesc(
            ad_name = jobAdRecord.name,
            exec_section = execSummary,
            project_list = [])

        oneResultList, usageStats = extractInfoFromJobAd(jobAdRecord, logger)
    #   oneResultList, usageStats = fakeExtractInfoFromJobAd()

        if not oneResultList:
            msg = f"Internal error on extracting activities from job description"
            self.workerError(logger, fileName, context, msg)
            return

        context['extracted'] = oneResultList.results_list
        if usageStats:
            context["llmrequests"] += usageStats.requests
            context["llmrequesttokens"] += usageStats.request_tokens
            context["llmresponsetokens"] += usageStats.response_tokens

        end = time.time()

        if usageStats:
            msg = f"Extracted {len(oneResultList.results_list)} activities from job description. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
        else:
            msg = f"Extracted {len(oneResultList.results_list)} activities from job description. {(end-start):9.4f} seconds."
        self.workerSnapshot(logger, fileName, context, msg)

        #--------------stage mapping

        start = time.time()

        context["stage"] = "mapping"
        self.workerSnapshot(logger, fileName, context, None)

        # ChromaDB calls do not account for LLM usage
        oneResultList = mapToActivity(oneResultList, chromaActivity, logger)
        #oneResultList = fakeMapToActivity()

        context['mapped'] = oneResultList.results_list

        end = time.time()

        msg = f"Mapped {len(oneResultList.results_list)} activities to vector database. {(end-start):9.4f} seconds"
        self.workerSnapshot(logger, fileName, context, msg)

        #----------------stage projects

        startAllProjects = time.time()

        context["stage"] = "projects"
        self.workerSnapshot(logger, fileName, context, None)

        context['projects'] = []
        prjCount = 0
        for chromaQuery in oneResultList.results_list:

            if chromaQuery[:4] == "--- ":
                logger.info(f"!!!!----!!!!!---skipping '{chromaQuery}'")
                continue
            start = time.time()
            oneDesc, usageStats = makeProject(chromaQuery, chromaScenario, logger)
            if oneDesc:
                prjCount += 1
                allDescriptions.project_list.append(oneDesc)
                context['projects'].append(oneDesc.description)

                if usageStats:
                    context["llmrequests"] += usageStats.requests
                    context["llmrequesttokens"] += usageStats.request_tokens
                    context["llmresponsetokens"] += usageStats.response_tokens

                end = time.time()

                msg = f"Project # {prjCount}: {oneDesc.title}. {(end-start):9.4f} seconds. {usageStats.request_tokens} request tokens. {usageStats.response_tokens} response tokens."
                self.workerSnapshot(logger, fileName, context, msg)

        endAllProjects = time.time()

        msg = f"Created {len(context['projects'])} projects. {(endAllProjects-startAllProjects):9.4f} seconds"
        self.workerSnapshot(logger, fileName, context, msg)


        #--------------stage completed

        totalEnd = time.time()

        context["stage"] = "completed"
        msg = f"Processing completed. Total time {(totalEnd-totalStart):9.4f} seconds. {context["llmrequests"]} LLM requests. {context["llmrequesttokens"]} request tokens. {context["llmresponsetokens"]} response tokens."
        self.workerSnapshot(logger, fileName, context, msg)
