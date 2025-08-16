#
# Read in JSON summary from job description
# fill in resume (CV) summary
# create timeline
#
import sqlite3
import sys
import tomli
import json
from pathlib import Path

from typing import Union

from pydantic_core import from_json

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions

# local
from common import ConfigSingleton
from common import OpenFile

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

#
# return tuple [bool, list[str]]
# on errors return [False, None]
#
def readListOfJobSummaries() -> tuple[bool, Union[list[str], str]] :
    dataPath = ConfigSingleton().conf["input_folder"]
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        return False, f"***ERROR: data path is not a folder: {folder_path}"
    fileNames = list(folder_path.glob("*.json"))
    for file in fileNames:
        boolResult, sourceStr = OpenFile.open(filePath = file, readContent = False)
        if not boolResult:
            return False, sourceStr
    return True, fileNames

def makeCVSummary(llmResult: LLMresult) -> tuple[bool, str] :
    # combine all keywords into one list
    totalMashup = []
    for key in llmResult.dict_of_results:
        totalMashup = totalMashup + llmResult.dict_of_results[key].results
    prompt = f"rewrite following JSON list as a paragraph for resume summary. Output only the answer. {totalMashup}"

    ollama_model = OpenAIModel(
        model_name='llama3.1:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
    )
    agent = Agent(ollama_model, retries=1, output_retries=1)

    try:
        result = agent.run_sync(prompt)
        return True, result.output
    except pydantic_ai.exceptions.UnexpectedModelBehavior:
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

    boolResult, listFilePathsOrError = readListOfJobSummaries()
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
