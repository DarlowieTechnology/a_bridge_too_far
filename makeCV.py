#
# Read in JSON summary from job description
# fill in resume (CV) summary
# create timeline
#
import sqlite3
import sys
import tomli
import json
import re
from pathlib import Path

from typing import Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions

#import asyncio


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile

# ---------------data types

class OneResultList(BaseModel):
    """represents one results from LLM call"""
    results: list[str]

class LLMresult(BaseModel):
    """represents collection of results from LLM calls"""
    originFile: str = ""
    dict_of_results: dict[str, OneResultList]

class FullResume(BaseModel):
    """Represents full resume (CV)"""
    originFile: str = ""
    summary: str = ""

# ---------------

def makeCVSummary(llmResult: LLMresult) -> tuple[bool, str] :
    # combine all keywords into one list
    totalMashup = []
    for key in llmResult.dict_of_results:
        if key != "questions":
            totalMashup = totalMashup + llmResult.dict_of_results[key].results
    prompt = f"rewrite following JSON list as a paragraph for resume summary. Output only the answer.\n {totalMashup}"

    ollamaModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollamaModel, 
                    output_type=OneResultList, 
                    retries=1, 
                    output_retries=1
                )
    try:
        result = agent.run_sync(prompt)
        return True, result.output
    except Exception as e:
        return False, f"ERROR: exception {e}"
    

def readAllScenarios(llmResult: LLMresult) -> tuple[bool, str] :
    # read scenarios
    array_results = []
    with sqlite3.connect(ConfigSingleton().conf["database_name"]) as conn:
        cur = conn.cursor()
        try:
            cur.execute("SELECT plot from SCENARIO;")
            rows = cur.fetchall()
            array_results = [str(row[0]) for row in rows]
        except sqlite3.IntegrityError as e:
            return False, f"sqlite3.IntegrityError, exception {e}"
        except Exception as e:
            return False, f"exception {e}"
    return True, array_results

# Seek adds question in known fixed format after this tag:
# ^Employer questions$
# ^Your application will include the following questions:$
#
def processQuestions(jobDescription:str) :
    oneResult = OneResultList(results = [])
    pattern = r'^Employer questions.*\nYour application will include the following questions:.*\n'
    matchQuestion = re.search(pattern, jobDescription, re.MULTILINE)
    if matchQuestion:
        questionsStr = jobDescription[matchQuestion.end(0):]
        oneResult.results = questionsStr.splitlines(False)
    return oneResult


def makeJobSummary(jobAdName : str, promptList : AllRecords) -> tuple[bool, Union[str, LLMresult]] :

    systemPrompt = f"""
        You are an expert in extraction of data from free form text.
        Your job is to extract the list requested by user prompt. 
        
        Here is the JSON schema for the result list model you must 
        use as context for what information is expected:
        {json.dumps(OneResultList.model_json_schema(), indent=2)}
        """

    ollamaModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
    agent = Agent(ollamaModel, 
                    output_type=OneResultList, 
                    retries=1, 
                    output_retries=1,
                    system_prompt = systemPrompt,
                )


    print(f"Processing {jobAdName}")

    llmResult = LLMresult(originFile = str(jobAdName), dict_of_results = {})

    boolResult, contentJDOrError = OpenFile.open(filePath = jobAdName, readContent = True)
    if not boolResult:
        return False, contentJDOrError

    for promptTemplate in promptList.list_of_records:
        try:
            prompt = promptTemplate.description + "\n" + contentJDOrError
            result = agent.run_sync(prompt)
            oneResultObject = OneResultList.model_validate_json(result.output.model_dump_json())
            # print(oneResultObject.model_dump_json(indent=2))
            llmResult.dict_of_results[promptTemplate.name] = oneResultObject
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            print(f"Skipping due to exception: {promptTemplate.name} : []")
            llmResult.dict_of_results[promptTemplate.name] = OneResultList(results = [])

    llmResult.dict_of_results["questions"] = processQuestions(jobDescription = contentJDOrError)
    # output summary in JSON alongside the original job description
    jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobAdName).stem + ".json"
    with open(jobDescriptionJSON, "w") as jsonOut:
        jsonOut.writelines(llmResult.model_dump_json(indent=2))

    return True, llmResult


def processJobSummary(llmResult : LLMresult) -> tuple[bool, str] :

    # read scenarios from database
    boolResult, scenarioOrError = readAllScenarios(llmResult)
    if not boolResult:
        return False, scenarioOrError

    oneResult = llmResult.dict_of_results["activity"]
    listActivities = oneResult.results
    for oneActivity in listActivities:

        print(f"Activity:{oneActivity}")
        print("--------------------------")

        prompt = f"chose all elements of following JSON list that match this activity: {oneActivity}. Output expanded paragraph for each match. {scenarioOrError}"

        ollama_model = OpenAIModel(
            model_name='llama3.1:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
        )
        agent = Agent(ollama_model, retries=1, output_retries=1)

        try:
            result = agent.run_sync(prompt)
            print(f"{result.output}\n\n")
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            print(f"ERROR: exception {e}")
        return

def main():

    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return

    try:
        with open(sys.argv[1], mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return

    # read list of TXT files - job ads
    boolResult, listFilePathsOrError = OpenFile.readListOfFileNames(
        ConfigSingleton().conf["input_folder"], 
        "*.txt"
    )
    if (not boolResult):
        print(listFilePathsOrError)
        return
    print(f"Found {len(listFilePathsOrError)} input files")

    # read prompts for data extraction
    boolResult, templateListOrError = OpenFile.readRecordJSON(
            ConfigSingleton().conf["prompt_folder"], 
            ConfigSingleton().conf["prompts"]
    )
    if (not boolResult):
        print(templateListOrError)
        return
    print(f"Found {len(templateListOrError)} prompt templates")


    for jobDescriptionPath in listFilePathsOrError:
        boolResult, llmResultOrError  = makeJobSummary(jobDescriptionPath)
        if not boolResult:
            print(llmResultOrError)
            return
        boolResult, summaryOrError  = makeCVSummary(llmResultOrError)
        if not boolResult:
            print(summaryOrError)
            return
        llmResultOrError.dict_of_results["CVsummary"] = OneResultList(results = [summaryOrError])


    return

    # read JSON summaries
    boolResult, listFilePathsOrError = OpenFile.readListOfFileNames(ConfigSingleton().conf["input_folder"], "*.json")


    if (not boolResult):
        print(listFilePathsOrError)
        return

    print(f"Found {len(listFilePathsOrError)} summary files")
    for jobSummaryPath in listFilePathsOrError:
        print(f"Processing {jobSummaryPath}")

        # read JSON summary
        boolResult, contentJDorError = OpenFile.open(filePath = jobSummaryPath, readContent = True)
        if not boolResult:
            print(contentJDorError)
            continue

        # make LLM result class out of JSON summary
        llmResult = LLMresult.model_validate(from_json(contentJDorError, allow_partial=False))

        # read scenarios from database
        boolResult, scenarioOrError = readAllScenarios(llmResult)
        if not boolResult:
            print(scenarioOrError)
            continue

        oneResult = llmResult.dict_of_results["activity"]
        listActivities = oneResult.results
        for oneActivity in listActivities:

            print(f"Activity:{oneActivity}")
            print("--------------------------")

            prompt = f"chose all elements of following JSON list that match this activity: {oneActivity}. Output expanded paragraph for each match. {scenarioOrError}"

            ollama_model = OpenAIModel(
                model_name='llama3.1:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
            )
            agent = Agent(ollama_model, retries=1, output_retries=1)

            try:
                result = agent.run_sync(prompt)
                print(f"{result.output}\n\n")
            except pydantic_ai.exceptions.UnexpectedModelBehavior:
                print(f"ERROR: exception {e}")
 
        return

        boolResult, summaryOrError = makeCVSummary(llmResult)
        if not boolResult:
            print(summaryOrError)
            continue
        llmResult.dict_of_results["CVsummary"] = OneResultList(results = [summaryOrError])

        # output resume in JSON alongside the original JSON description
        jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobSummaryPath).stem + ".resume.json"
        with open(jobDescriptionJSON, "w") as jsonOut:
            jsonOut.writelines(llmResult.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
#    asyncio.run(main())
