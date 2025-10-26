from django.shortcuts import render
from django.http import JsonResponse

from typing import List

import chromadb
from chromadb import Collection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb import QueryResult

import pydantic_ai
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

import tomli
import logging
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import threading


from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import genai_prices

# local
sys.path.append("..")
sys.path.append("../..")

from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile
from common import DebugUtils, OneDesc, AllDesc, OneResultList, OneEmployer, AllEmployers

from generator_workflow import GeneratorWorkflow




def status(request):

    context = {}

    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger(request.session.session_key)

    statusFileName = "status." + request.session.session_key + ".json"
    try:
        with open(statusFileName, "r") as jsonIn:
            context = json.load(jsonIn)
    except Exception as e:
        errorMsg = f"Status Page: status file error {e}"
        logger.info(errorMsg)
        context['status'] = errorMsg
        return JsonResponse(context)
    
    msg = f"Status: Opened {statusFileName}"
    logger.info(msg)
    return JsonResponse(context)


def index(request):
    # create session key and log per session
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger(request.session.session_key)
    logger.info(f"Starting session")

    return render(request, "generator/index.html", None)

def process(request):

    context = {}

    if not request.session.session_key:
        request.session.create() 

    logger = logging.getLogger(request.session.session_key)

    logger.info(f"Process: Serving POST")
    statusFileName = "status." + request.session.session_key + ".json"
    boolResult, sessionInfoOrError = OpenFile.open(statusFileName, True)
    if boolResult:
        contextOld = json.loads(sessionInfoOrError)
        logger.info("Process: Existing async processing found")
        if contextOld["stage"] in ["error", "completed"]:
            logger.info("Process: Removing completed session file")
            pass
        else:    
            return render(request, "generator/process.html", context)

    thread = threading.Thread(
        target=threadWorker, 
        args=(request.session.session_key, statusFileName, request.POST['adtext']))
    thread.start()

    msg = f"Processing..."
    context['status'] = msg
    context['stage'] = 'starting'
    logger.info(msg)
    with open(statusFileName, "w") as jsonOut:
        json.dump(context, jsonOut)
    return render(request, "generator/process.html", context)



def extractExecSection(jobInfo : OneRecord, 
                       logger : logging.Logger, 
                       fileName : str, 
                       context : dict) -> tuple[ OneDesc, Usage] :

    systemPromptPhase1 = f"""
        You are an expert in cyber security, information technology and software development 
        You will be supplied text of job advertisement.
        Your job is to extract information from the text that matches user's request.
        Here is the JSON schema for the OneDesc model you must use as context for what information is expected:
        {json.dumps(OneDesc.model_json_schema(), indent=2)}
        """
    try:
        ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    except Exception as e:
        msg = f"OpenAIModel API exception: {e}"
        workerError(logger, fileName, context, msg)
        return None, None

    try:
        agent = Agent(ollModel, 
                output_type=OneDesc,                  
                system_prompt = systemPromptPhase1,
                retries=10,
                output_retries=10)
    except Exception as e:
        msg = f"Agent creation exception: {e}"
        workerError(logger, fileName, context, msg)
        return None, None

    promptPhase1 = f"""Extract the required role suitable for CV from the text below.
    Make required role the title.
    Fill description with generic description of the required role in past tense suitable for CV.
    Do not add formatting.
    Output only the result.
    \n
    {jobInfo.description}"""

    try:
        result = agent.run_sync(promptPhase1)
        oneDesc = OneDesc.model_validate_json(result.output.model_dump_json())
        oneDesc.description = oneDesc.description.replace("\n", " ")
        oneDesc.description = oneDesc.description.encode("ascii", "ignore").decode("ascii")
        DebugUtils.logPydanticObject(oneDesc, "Executive Summary", logger)
        runUsage = result.usage()
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        msg = f"LLM summary extraction exception: {e}"
        workerError(logger, fileName, context, msg)
        return None, None
    except Exception as e:
        msg = f"Summary extraction exception: {e}"
        workerError(logger, fileName, context, msg)
        return None, None
    return oneDesc, runUsage


def extractInfoFromJobAd(jobInfo : OneRecord, logger : logging.Logger) -> tuple[ OneResultList, Usage] :

    systemPromptPhase1 = f"""
        You are an expert in cyber security, information technology and software development 
        You will be supplied text of job advertisement.
        Your job is to extract information from the text that matches user's request.
        """

    ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollModel, 
                system_prompt = systemPromptPhase1,
                retries=5,
                output_retries=5)

    promptPhase1 = f"""Extract the list of 
    activities, technologies, methodologies, software services, and software products from the text below.
    Combine all items in common list of strings.
    Do not separate by category.
    Avoid single word items.
    Output only lower-case characters.
    Output only the result.
    \n
    {jobInfo.description}
            """

    try:
        result = agent.run_sync(promptPhase1)
        runUsage = result.usage()
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        logger.info(f"extractInfoFromJobAd: phase 1 exception: {e}")
        return None, None

    phase2Input = result.output

    systemPromptPhase2 = f"""
        You are an expert in JSON processing.
        Input prompt contains list in JSON format. 

        Here is the JSON schema for the OneResultList model you must use as context for what information is expected:
        {json.dumps(OneResultList.model_json_schema(), indent=2)}
        """
    promptPhase2 = f"""
                {phase2Input}
            """

    agentPhase2 = Agent(ollModel, 
                output_type=OneResultList,
                system_prompt = systemPromptPhase2,
                retries=5,
                output_retries=5)
    try:
        result = agentPhase2.run_sync(promptPhase2)
        oneResultList = OneResultList.model_validate_json(result.output.model_dump_json())
        runUsage = runUsage + result.usage()
        DebugUtils.logPydanticObject(oneResultList, "list from job ad", logger)
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        logger.info(f"extractInfoFromJobAd: phase 2 exception: {e}")
        return None, None
    return oneResultList, runUsage


def mapToActivity(oneResultList : OneResultList, chromaDBCollection : Collection, logger : logging.Logger) -> OneResultList :
    itemSet = set()
    for itm in oneResultList.results_list:
        listNew = getChromaDBMatchActivity(chromaDBCollection, itm, logger)
        if not len(listNew.results_list):
            itemSet.add(f"--- No match for activity `{itm}` ---")
        else:
            for itemNew in listNew.results_list:
                itemSet.add(itemNew)
    
    oneResultList = OneResultList(results_list = list(itemSet))
    DebugUtils.logPydanticObject(oneResultList, "Mapped list", logger)
    return oneResultList


def getChromaDBMatchActivity(chromaDBCollection : Collection, queryString : str, logger : logging.Logger) -> OneResultList :

    totals = set()

    queryResult = chromaDBCollection.query(query_texts=[queryString], n_results=1)
    cutDist = ConfigSingleton().conf["rag_distmatch"]
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
        logger.info(f"Query {queryString} did not get matches less than {ConfigSingleton().conf['rag_distmatch']}")
        return OneResultList(results_list = [])

    return OneResultList(results_list=list(totals))


def makeProject(chromaQuery : str, chromaScenario : Collection, logger : logging.Logger) -> tuple [OneDesc, Usage] :

    logger.info(f"Scenario query: ({chromaQuery})")
    queryResult = chromaScenario.query(query_texts=[chromaQuery], n_results=3)

    idx = -1
    numberChosen = 0
    distList = []
    combinedDoco = ""
    for distFloat in queryResult["distances"][0] :
        idx += 1
        if (distFloat > ConfigSingleton().conf["rag_scenario"]) : 
            break

        oneResult = OneQueryResult(
                id = queryResult["ids"][0][idx],
                name = queryResult["metadatas"][0][idx]["docName"], 
                desc = queryResult["documents"][0][idx], 
                query = chromaQuery,
                distance=distFloat 
            )
#            DebugUtils.logPydanticObject(oneResult, "one query result", logger)

        distList.append(distFloat)
        docText = ""
        if (queryResult["documents"]) :
            docText = queryResult["documents"][0][idx]
        metaInf = ""
        if (queryResult["metadatas"]) :
            metaInf = queryResult["metadatas"][0][idx]["docName"]
        combinedDoco = combinedDoco + "\n" + docText
        numberChosen = numberChosen + 1

    if not numberChosen :
        logger.info(f"ERROR: cannot find ChromaDB records under distance {ConfigSingleton().conf['rag_scenario']}")
        return None, None
    
    logger.info(f"Selected {numberChosen} scenarios from ChromaDB. Distances: {distList}")

    systemPrompt = f"""
        You are an expert technical writer. 
        Create title and description of the project based on information supplied. 
        Create a full paragraph for description.
        Do not format text. Remove line feeds and carriage returns.
        Output only the result.
        Here is the JSON schema for the OneDesc model you must use as context for what information is expected:
        {json.dumps(OneDesc.model_json_schema(), indent=2)}
        """
    prompt = f"{combinedDoco}"
    logger.info(f"-------prompt---------\n{prompt}\n-------------\n")

    ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    
    agent = Agent(ollModel,
                output_type=OneDesc,
                system_prompt = systemPrompt,
                retries=5,
                output_retries=5)
    try:
        result = agent.run_sync(prompt)
        oneDesc = OneDesc.model_validate_json(result.output.model_dump_json())
        oneDesc.description = oneDesc.description.replace("\n", "")
        runUsage = result.usage()
        DebugUtils.logPydanticObject(oneDesc, "Project Description", logger)
    except pydantic_ai.exceptions.UnexpectedModelBehavior:
        logger.info(f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior")
        return None, None
    return oneDesc, runUsage


def makeWordDoc(allDesc : AllDesc) :

    allEmployers = AllEmployers(employer_list = [])
    allEmployers.employer_list.append(OneEmployer(
        name = "Darlowie Security Consulting",
        blurb = "Principal Consultant",
        start = datetime(2024, 11, 1),
        end = datetime.now(),
        numJobs = 3
    ))
    allEmployers.employer_list.append(OneEmployer(
        name = "NCC Group APAC",
        blurb = "Principal Consultant",
        start = datetime(2014, 12, 1),
        end = datetime(2024, 10, 30),
        numJobs = 15
    ))
    allEmployers.employer_list.append(OneEmployer(
        name = "NCC Group USA",
        blurb = "Senior Consultant",
        start = datetime(2013, 5, 15),
        end = datetime(2014, 10, 30),
        numJobs = 10
    ))

    # Create a new Document
    doc = Document()

    title = doc.add_heading(level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    titleRun = title.add_run('Anton Pavlov')
    font = titleRun.font
    font.Name = 'Cambria'
    font.size = Pt(20)

    info_para = doc.add_paragraph()
    info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info_para.add_run('Melbourne, Victoria  Ph: 0428065430  Email:XXXX@YYYY.com')

    exec_paragraph = doc.add_paragraph()
    exec_paragraph.add_run(allDesc.exec_section.title).bold = True
    exec_paragraph.add_run('\n')
    exec_paragraph.add_run(allDesc.exec_section.description)

    expHeading = doc.add_heading(level=2)
    expHeading.add_run('Experience')

    idxCurrentJob = 0
    idxGlobalJob = 0
    for emp in allEmployers.employer_list:
        date_name_para = doc.add_paragraph()
        date_name_para.add_run(emp.start.strftime('%b %Y') + '-' +  emp.end.strftime('%b %Y'))
        date_name_para.add_run(' ')
        date_name_para.add_run(emp.name).bold = True
        blurb_para = doc.add_paragraph()
        blurb_para.add_run(emp.blurb)
        for oneDesc in allDesc.project_list[idxGlobalJob:] :
            idxCurrentJob += 1
            if (idxCurrentJob > emp.numJobs) :
                idxCurrentJob = 0
                break
            list_paragraph = doc.add_paragraph(style='List Bullet')
            list_paragraph.add_run(oneDesc.title).bold = True
            list_paragraph.add_run('\n')
            list_paragraph.add_run(oneDesc.description)
            list_paragraph.add_run('\n')
            idxGlobalJob += 1

    # Save the document
    doc.save(allDesc.ad_name + ".resume.docx")


def fakeExtractExecSection() ->  tuple[OneDesc, Usage] :

    oneDesc = OneDesc(title='Senior Cyber Security Advisor', 
                      description='Utilized senior expertise in enterprise-wide security outcomes through strategic engagement and technical leadership. Established strong relationships with stakeholders and provided expert guidance to enable secure business missions.')
    return oneDesc, None

def fakeExtractInfoFromJobAd() -> tuple [OneResultList, Usage] :
    return OneResultList( results_list = [
        "strategic relationship management",
        "cybersecurity advocacy",
        "security assessment engagement",
        "operational efficiency",
        "end-to-end engagement leadership",
        "risk analysis & treatment advisory",
        "security risk assurance coordination",
        "enterprise security guidance",
        "business process reengineering",
        "operational process development and continuous improvement",
        "project management, multitasking, and organizational skills",
        "comptia security+",
        "(isc)� sscp - systems security certified practitioner",
        "isaca cybersecurity fundamentals",
        "giac security essentials (gsec)",
        "microsoft sc-900 - security, compliance, and identity fundamentals",
        "cisco certified cyberops associate",
        "global network & technology - security & operations",
        "telstra's assets and infrastructure"
    ]), None

def fakeMapToActivity() ->  tuple [OneResultList, Usage] :
    return OneResultList( results_list = [
        "Demonstrated project management skills.",
        "Managed the coordination and evaluation of security assurance activities.",
        "Participated in continuous process improvement",
        "Acquired Offensive Security Certified Professional (OSCP) certification.",
        "Conducted risk assessments and developed risk mitigation plans.",
        "Engaged with global teams for security operations.",
        "Supported the creation and governance of enterprise security strategies and standards.",
        "Acquired Comptia Security+ certification.",
        "Acquired Certified in Risk and Information Systems Control (CRISC) certification from ISACA.",
        "Performed customer relationship management.",
        "Conducted security threat assessments and risk assessments.",
        "Delivered end-to-end engagements.",
        "Led Cybersecurity initiatives.",
        "Acquired Certified Cryptoasset Anti-Financial Crime Specialist (CCAS) certification."
    ]), None


# use this to display genAI pricing
#   
def results(request):

    providers = [
        { "provider" : "anthropic", "model": "claude-3-5-haiku-latest"  },
        { "provider" : "aws", "model": "nova-pro-v1" },
        { "provider" : "azure", "model": "gpt-4" },
        { "provider" : "deepseek", "model": "deepseek-chat" },
        { "provider" : "google", "model": "gemini-pro-1.5" },
        { "provider" : "openai", "model": "gpt-4" },
        { "provider" : "openrouter", "model": "gpt-4" },
        { "provider" : "x-ai", "model": "grok-3" },
        { "provider" : "x-ai", "model": "grok-4-0709" }
    ]

    context = {}
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger(request.session.session_key)
    logger.info(f"Results: Serving GET")

    context["totalrequests"] = request.GET["totalrequests"]
    context["totalrequesttokens"] = request.GET["totalrequesttokens"]
    context["totalresponsetokens"] = request.GET["totalresponsetokens"]

    context["llminfo"] = []
    for providerInfo in providers:
        price_data = genai_prices.calc_price(
            genai_prices.Usage(input_tokens=int(context["totalrequesttokens"]), output_tokens=int(context["totalresponsetokens"])),
            model_ref= providerInfo["model"],
            provider_id = providerInfo["provider"]
        )
        item = {}
        item["provider"] = providerInfo["provider"]
        item["model"] = providerInfo["model"]
        item["costusd"] = f"{price_data.total_price:.4f}"
        audValue = float(price_data.total_price) * 1.53
        item["costaud"] = f"{audValue:.4f}"

        context["llminfo"].append(item)

    return render(request, "generator/results.html", context)

