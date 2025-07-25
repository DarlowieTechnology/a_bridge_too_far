#
# Read in JSON summary from job description
# Match products/technologies/role/certifications to known objects
#
import sqlite3
import sys
import glob
import tomli
import json
from pathlib import Path

from pydantic_core import from_json

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions


# local
from common import ConfigSingleton

# ---------------data types

class OneResultList(BaseModel):
    """represents one results from LLM call"""
    results: list[str]

class LLMresult(BaseModel):
    """represents collection of results from LLM calls"""
    originFile: str = ""
    dict_of_results: dict[str, OneResultList]


#
# open text file
# return tuple [bool, content]
# if error, return [False, None]
#
def openTextFile(filePath : str, readContent : bool) -> tuple[bool, str] :
    file_path = Path(filePath)
    if not file_path.is_file():
        print(f"***ERROR: Error opening file {filePath}")
        return False, None
    try:
        with open(filePath, "r") as textFile:
            if readContent:
                return True, textFile.read()
            else:
               return True, None
    except FileNotFoundError as e:
        print(f"***ERROR: Error opening file {filePath}, exception {e}")
    except PermissionError as e:
        print(f"***ERROR: Permission error opening file {filePath}, exception {e}")
    except Exception as e:
        print(f"***ERROR: General error opening file {filePath}, exception {e}")
    return False, None

#
# return tuple [bool, list[str]]
# on errors return [False, None]
#
def readListOfJobSummaries() -> tuple[bool, list[str]] :
    dataPath = ConfigSingleton().conf["input_folder"]
    files: list[str] = []
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        print(f"***ERROR: data path is not a folder: {folder_path}")
        return False, None
    rawFiles = glob.glob(dataPath + "/*.json")
    for file in rawFiles:
        boolResult, sourceStr = openTextFile(filePath = file, readContent = False)
        if not boolResult:
            return False, None
        files.append(file)
    return True, files

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

    boolResult, listFilePaths = readListOfJobSummaries()
    if (not boolResult):
        print(f"***ERROR: Opening summary files in {ConfigSingleton().conf["input_folder"]}")
        return

    print(f"Found {len(listFilePaths)} summary files")
    for jobSummaryPath in listFilePaths:

        print(f"Processing {jobSummaryPath}")
        boolResult, contentJD = openTextFile(filePath = jobSummaryPath, readContent = True)
        if not boolResult:
            print(f"Skipping {jobSummaryPath}")
            continue
        llmResult = LLMresult.model_validate(from_json(contentJD, allow_partial=False))

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
            print(f"{result.output}")
        except pydantic_ai.exceptions.UnexpectedModelBehavior:
            print("ERROR")

if __name__ == "__main__":
    main()
