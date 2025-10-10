from django.shortcuts import render

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

import tomli
import logging
import json
import sys
from datetime import datetime

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

import threading

# local
sys.path.append("..")

from common import OneRecord, AllRecords, OneQueryResult, AllQueryResults, ConfigSingleton, OpenFile
from common import DebugUtils, OneDesc, AllDesc, OneResultList, OneEmployer, AllEmployers


def index(request):
    # create session key and log per session
    if not request.session.session_key:
        request.session.create() 
    logger = logging.getLogger(request.session.session_key)
    logger.info(f"Starting session")

    return render(request, "generator/index.html", None)

def process(request):

    context = {}

    logger = logging.getLogger(request.session.session_key)

    if request.method == "GET":
        logger.info(f"Serving GET")
        for key in ['status', 'jobtitle', 'adtext', 'execsummary', 'extracted', 'mapped', 'projects']:
            context[key] = request.session.get(key, "")
        return render(request, "generator/process.html", context)

    # start POST processing
    logger.info(f"Serving POST - cleaning session storage")
    for key in ['status', 'jobtitle', 'adtext', 'execsummary', 'extracted', 'mapped', 'projects']:
        request.session[key] = ""
    request.session.flush()

#    t = threading.Thread(target=doProcessing,args=[request], daemon=True)
#    t.start()

    context['status'] = "Started processing"
    request.session['status'] = context['status']
    logger.info(f"Starting processing thread")

    configName = "default.toml"
    try:
        with open(configName, mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        logger.info(f"***ERROR: Cannot open config file {configName}, exception {e}")
        request.session['status'] = 'Internal error on config file'
        return render(request, "generator/process.html", context)
    logger.info(f"Opened config file {configName}")

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().getAbsPath("rag_datapath"),
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    logger.info(f"Created ChromaDB client")

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
        logger.info(f"Collection {collectionName} opened with {chromaActivity.count()} documents")
    except Exception as e:
        logger.info(f"Exception: {e}")
        request.session['status'] = f'Internal error on ChromaDB {collectionName} collection'
        return render(request, "generator/process.html", context)
    logger.info(f"Opened ChromaDB collection {collectionName} with {chromaActivity.count()} documents ")

    collectionName = "scenario"
    try:
        chromaScenario = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
        logger.info(f"Collection {collectionName} opened with {chromaActivity.count()} documents")
    except Exception as e:
        logger.info(f"Exception: {e}")
        request.session['status'] = f'Internal error on ChromaDB {collectionName} collection'
        return render(request, "generator/process.html", context)
    logger.info(f"Opened ChromaDB collection {collectionName} with {chromaScenario.count()} documents ")

    jobAdRecord = OneRecord(
        id = "", 
        name=str(request.session.session_key), 
        description=request.POST['adtext']
    )
    logger.info(f"Copied job description from client POST data - {len(request.POST['adtext'])} bytes")
    request.session['adtext'] = request.POST['adtext']
    request.session.flush()
    context['adtext'] = request.POST['adtext']

    #execSummary = extractExecSection(jobAdRecord, logger)
    execSummary = fakeExtractExecSection()
    request.session['jobtitle'] = execSummary.title
    request.session['execsummary'] = execSummary.description
    request.session['status'] = 'Extracted executive summary'
    request.session.flush()
    logger.info(f"Extracted executive summary from ad text")

    context['jobtitle'] = execSummary.title
    context['execsummary'] = execSummary.description


    allDescriptions = AllDesc(
        ad_name = jobAdRecord.name,
        exec_section = execSummary,
        project_list = [])

    #oneResultList = extractInfoFromJobAd(jobAdRecord, logger)
    oneResultList = fakeExtractInfoFromJobAd()
    logger.info(f"Extracted activities from ad text")

    if not oneResultList:
        request.session['status'] = 'Internal error on extractInfoFromJobAd'
        return
    request.session['extracted'] = oneResultList.results_list
    request.session['status'] = 'Extracted activities'
    request.session.flush()

    context['extracted'] = oneResultList.results_list


    #oneResultList = mapToActivity(oneResultList, chromaActivity, logger)
    oneResultList = fakeMapToActivity()
    logger.info(f"Matched activities to Chroma DB")

    request.session['mapped'] = oneResultList.results_list
    request.session['status'] = 'Mapped activities'
    request.session.flush()

    context['mapped'] = oneResultList.results_list


    request.session['projects'] = []
    for chromaQuery in oneResultList.results_list:
        oneDesc = makeProject(chromaQuery, chromaScenario, logger)
        if oneDesc:
            allDescriptions.project_list.append(oneDesc)
#            request.session['projects'].append(oneDesc.description)
            request.session['status'] = 'Created project'
            request.session.flush()
    logger.info(f"Created projects")
    request.session['status'] = 'Processing completed'
    request.session.flush()

    context['mapped'] = allDescriptions.project_list
    return render(request, "generator/process.html", context)

def status(request):
    return render(request, "generator/status.html", None)


def results(request):
    return render(request, "generator/results.html", None)


# --------- long running thread func

def doProcessing(request): 

    logger = logging.getLogger(request.session.session_key)
    logger.info(f"Starting processing thread")

    configName = "default.toml"
    try:
        with open(configName, mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        logger.info(f"***ERROR: Cannot open config file {configName}, exception {e}")
        request.session['status'] = 'Internal error on config file'
        return
    logger.info(f"Opened config file {configName}")

    chromaClient = chromadb.PersistentClient(
        path=ConfigSingleton().getAbsPath("rag_datapath"),
        settings=Settings(anonymized_telemetry=False),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
    )
    logger.info(f"Created ChromaDB client")

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
        logger.info(f"Collection {collectionName} opened with {chromaActivity.count()} documents")
    except Exception as e:
        logger.info(f"Exception: {e}")
        request.session['status'] = f'Internal error on ChromaDB {collectionName} collection'
        return
    logger.info(f"Opened ChromaDB collection {collectionName} with {chromaActivity.count()} documents ")

    collectionName = "scenario"
    try:
        chromaScenario = chromaClient.get_collection(
            name=collectionName,
            embedding_function=ef
        )
        logger.info(f"Collection {collectionName} opened with {chromaActivity.count()} documents")
    except Exception as e:
        logger.info(f"Exception: {e}")
        request.session['status'] = f'Internal error on ChromaDB {collectionName} collection'
        return
    logger.info(f"Opened ChromaDB collection {collectionName} with {chromaScenario.count()} documents ")

    jobAdRecord = OneRecord(
        id = "", 
        name=str(request.session.session_key), 
        description=request.POST['adtext']
    )
    logger.info(f"Copied job description from client POST data - {len(request.POST['adtext'])} bytes")
    request.session['adtext'] = request.POST['adtext']
    request.session.flush()

    execSummary = extractExecSection(jobAdRecord, logger)
    # execSummary = fakeExtractExecSection()
    request.session['jobtitle'] = execSummary.title
    request.session['execsummary'] = execSummary.description
    request.session['status'] = 'Extracted executive summary'
    request.session.flush()
    logger.info(f"Extracted executive summary from ad text")

    allDescriptions = AllDesc(
        ad_name = jobAdRecord.name,
        exec_section = execSummary,
        project_list = [])

    oneResultList = extractInfoFromJobAd(jobAdRecord, logger)
    #oneResultList = fakeExtractInfoFromJobAd()
    logger.info(f"Extracted activities from ad text")

    if not oneResultList:
        request.session['status'] = 'Internal error on extractInfoFromJobAd'
        return
    request.session['extracted'] = oneResultList.results_list
    request.session['status'] = 'Extracted activities'
    request.session.flush()

    oneResultList = mapToActivity(oneResultList, chromaActivity, logger)
    #oneResultList = fakeMapToActivity()
    logger.info(f"Matched activities to Chroma DB")

    request.session['mapped'] = oneResultList.results_list
    request.session['status'] = 'Mapped activities'
    request.session.flush()

    request.session['projects'] = []
    for chromaQuery in oneResultList.results_list:
        oneDesc = makeProject(chromaQuery, chromaScenario, logger)
        if oneDesc:
            allDescriptions.project_list.append(oneDesc)
#            request.session['projects'].append(oneDesc.description)
            request.session['status'] = 'Created project'
            request.session.flush()

    logger.info(f"Created projects")
    request.session['status'] = 'Processing completed'
    request.session.flush()







def extractExecSection(jobInfo : OneRecord, logger : logging.Logger) -> OneDesc :

    systemPromptPhase1 = f"""
        You are an expert in cyber security, information technology and software development 
        You will be supplied text of job advertisement.
        Your job is to extract information from the text that matches user's request.
        Here is the JSON schema for the OneDesc model you must use as context for what information is expected:
        {json.dumps(OneDesc.model_json_schema(), indent=2)}
        """

    ollModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollModel, 
                output_type=OneDesc,                  
                system_prompt = systemPromptPhase1,
                retries=10,
                output_retries=10)

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
        oneDesc.description = oneDesc.description.replace("\n", "")
        DebugUtils.logPydanticObject(oneDesc, "Executive Summary", logger)
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        logger.info(f"extractExecSection: Skipping due to exception: {e}")
        return None
    return oneDesc


def extractInfoFromJobAd(jobInfo : OneRecord, logger : logging.Logger) -> OneResultList :

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
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        logger.info(f"extractInfoFromJobAd: phase 1 exception: {e}")
        return None

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
        DebugUtils.logPydanticObject(oneResultList, "list from job ad", logger)
    except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
        logger.info(f"extractInfoFromJobAd: phase 2 exception: {e}")
        return None
    return oneResultList


def mapToActivity(oneResultList : OneResultList, chromaDBCollection : Collection, logger : logging.Logger) -> OneResultList :
    itemSet = set()
    for itm in oneResultList.results_list:
        listNew = getChromaDBMatchActivity(chromaDBCollection, itm, logger)
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
        logger.info(f"Query {queryString} did not get matches less than {ConfigSingleton().conf["rag_distmatch"]}")
        return OneResultList(results_list = [])

    return OneResultList(results_list=list(totals))


def makeProject(chromaQuery : str, chromaScenario : Collection, logger : logging.Logger) -> OneDesc :

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
        logger.info(f"ERROR: cannot find ChromaDB records under distance {ConfigSingleton().conf["rag_scenario"]}")
        return None
    
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
        DebugUtils.logPydanticObject(oneDesc, "Project Description", logger)
    except pydantic_ai.exceptions.UnexpectedModelBehavior:
        logger.info(f"Exception: pydantic_ai.exceptions.UnexpectedModelBehavior")
        return None
    return oneDesc


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


def fakeExtractExecSection() ->  OneDesc :

    oneDesc = OneDesc(title='Senior Cyber Security Advisor', 
                      description='Utilized senior expertise in enterprise-wide security outcomes through strategic engagement and technical leadership. Established strong relationships with stakeholders and provided expert guidance to enable secure business missions.')
    return oneDesc

def fakeExtractInfoFromJobAd() -> OneResultList :
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
    ])

def fakeMapToActivity() ->  OneResultList :
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
    ])
  
