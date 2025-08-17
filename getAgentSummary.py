import sys
import tomli
import json
import re
import sqlite3
from pathlib import Path

from typing import Union

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions

import asyncio


# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile

#--------------------------------------------------------

class PromptTemplate(BaseModel):
    """represents prompt template with named parameters."""
    name: str = ""
    list: list[str]
    value: str = ""

class OneResultList(BaseModel):
    """represents one results from LLM call"""
    results: list[str]

class LLMresult(BaseModel):
    """represents collection of results from LLM calls"""
    originFile: str = ""
    dict_of_results: dict[str, OneResultList]

#-------------------------------------------------------

#
# return tuple [bool, list[str]]
# on errors return [False, Error]
#
def readListOfFileNames() -> tuple[bool, Union[list[str],str]] :
    dataPath = ConfigSingleton().conf["input_folder"]
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        return False, f"***ERROR: data path is not a folder: {folder_path}"
    fileNames = list(folder_path.glob("*.txt"))
    for file in fileNames:
        boolResult, sourceStr = OpenFile.open(filePath = file, readContent = False)
        if not boolResult:
            return False, [sourceStr]
    return True, fileNames


def fillListFromDatabase(tableName:str) -> list[str]:
    if (tableName == "position"):
            with sqlite3.connect(ConfigSingleton().conf["database_name"]) as conn:
                cur = conn.cursor()
                try:
                    cur.execute("SELECT name from POSITION;")
                    rows = cur.fetchall()
                    array_results = [str(row[0]) for row in rows]
                    return array_results
                except sqlite3.IntegrityError as e:
                    print(f"sqlite3.IntegrityError, exception {e}")
                except Exception as e:
                    print(f"exception {e}")
    return []

#
# read prompts from JSON file
# return tuple [bool, list[PromptTemplate]]
# on error return [False, Error]
#
def readPrompts() -> tuple[bool, Union[list[PromptTemplate],str]] :
    promptFile = ConfigSingleton().conf["prompts"]
    boolResult, sourceStr = OpenFile.open(filePath = promptFile, readContent = True)
    if not boolResult:
        return False, sourceStr
    try:
        sourceJson = json.loads(sourceStr)
        list_of_templates: list[PromptTemplate] = []
        for prompt in sourceJson:
            prompt["list"] = fillListFromDatabase(prompt["name"])
            list_of_templates.append(PromptTemplate(**prompt))
        return True, list_of_templates
    except json.JSONDecodeError as e:
        return False, f"***ERROR: file {promptFile}, exception {e}"
    except Exception as e:
        return False, f"***ERROR: file {promptFile}, exception {e}"

def formPrompt(prompt : PromptTemplate) -> str:
    
    if not len(prompt.list) :
        return prompt.value
    
    return prompt.value + " (" + ",".join(prompt.list)  + ")"


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


async def main():
    
    if len(sys.argv) < 2:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG\nExample: {sys.argv[0]} default.toml")
        return
    try:
        with open(sys.argv[1], mode="rb") as fp:
                ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return

    boolResult, templateListOrError = readPrompts()
    if (not boolResult):
        print(templateListOrError)
        return

    boolResult, listFilePathsOrError = readListOfFileNames()
    if (not boolResult):
        print(listFilePathsOrError)
        return

    print(f"Found {len(listFilePathsOrError)} input files")
    for jobDescriptionPath in listFilePathsOrError:

        print(f"Processing {jobDescriptionPath}")
        llmResult = LLMresult(originFile = str(jobDescriptionPath), dict_of_results = {})
        boolResult, contentJDOrError = OpenFile.open(filePath = jobDescriptionPath, readContent = True)
        if not boolResult:
            print(contentJDOrError)
            continue

        for promptTemplate in templateListOrError:

            ollamaModel = OpenAIModel(model_name=ConfigSingleton().conf["main_llm_name"], 
                                provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"]))
            agent = Agent(ollamaModel, output_type=OneResultList, retries=1, output_retries=1)

            @agent.system_prompt
            async def system_prompt() -> str:
                return "Given the following text in square brackets, \
                    your job is to output information that matches user's request.\
                    Reply with the answer only. Format answer as JSON list. [" + contentJDOrError + "]"
            try:
                promptString = formPrompt(promptTemplate)
                print(promptString)
                print("----------------")
                result = agent.run_sync(promptString)
                print(f"{promptTemplate.name} : {result.output}")
                llmResult.dict_of_results[promptTemplate.name] = result.output
            except pydantic_ai.exceptions.UnexpectedModelBehavior:
                print(f"Skipping due to exception: {promptTemplate.name} : []")
                llmResult.dict_of_results[promptTemplate.name] = OneResultList(results = [])

        llmResult.dict_of_results["questions"] = processQuestions(jobDescription = contentJDOrError)

        # output summary in JSON alongside the original job description
        jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobDescriptionPath).stem + ".json"
        with open(jobDescriptionJSON, "w") as jsonOut:
            res = llmResult.model_dump_json(indent=2)
            jsonOut.writelines(res)


if __name__ == "__main__":
    asyncio.run(main())
