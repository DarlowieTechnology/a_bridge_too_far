import sys
import glob
import tomli
import json
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions

# local
from common import ConfigSingleton


class PromptTemplate(BaseModel):
    """represents prompt template with named parameters"""
    name: str = ""
    value: str = ""

class OneResultList(BaseModel):
    """represents one results from LLM call - pass this class in AI agent as a template"""
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
def readListOfJobs() -> tuple[bool, list[str]] :
    dataPath = ConfigSingleton().conf["input_folder"]
    files: list[str] = []
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        print(f"***ERROR: data path is not a folder: {folder_path}")
        return False, None
    rawFiles = glob.glob(dataPath + "/*.txt")
    for file in rawFiles:
        boolResult, sourceStr = openTextFile(filePath = file, readContent = False)
        if not boolResult:
            return False, None
        files.append(file)
    return True, files


#
# read prompts from JSON file
# return tuple [bool, list[PromptTemplate]]
# on error return [False, None]
#
def readPrompts() -> tuple[bool, list[PromptTemplate]] :
    promptFile = ConfigSingleton().conf["prompts"]
    list_of_templates: list[PromptTemplate] = []
    boolResult, sourceStr = openTextFile(filePath = promptFile, readContent = True)
    if not boolResult:
        print("***ERROR: no prompts for LLM found")
        return False, None
    sourceJson = json.loads(sourceStr)
    for prompt in sourceJson:
        list_of_templates.append(PromptTemplate(
                name=prompt["name"], 
                value=prompt["value"]
            ))

    return True, list_of_templates



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

    boolResult, templateList = readPrompts()
    if (not boolResult):
        print(f"***ERROR: Cannot open prompt file {ConfigSingleton().conf["prompts"]}")
        return

    boolResult, listFilePaths = readListOfJobs()
    if (not boolResult):
        print(f"***ERROR: Opening input files in {ConfigSingleton().conf["input_folder"]}")
        return

    print(f"Found {len(listFilePaths)} input files")
    for jobDescriptionPath in listFilePaths:

        print(f"Processing {jobDescriptionPath}")
        llmResult = LLMresult(originFile = jobDescriptionPath, dict_of_results = {})
        boolResult, contentJD = openTextFile(filePath = jobDescriptionPath, readContent = True)
        if not boolResult:
            print(f"Skipping {jobDescriptionPath}")
            continue

        for promptTemplate in templateList:
            ollama_model = OpenAIModel(
                model_name='llama3.1:latest', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
            )
            agent = Agent(ollama_model, output_type=OneResultList, retries=1, output_retries=1)

            @agent.system_prompt
            async def system_prompt() -> str:
                strOut = f"""\
            Given the following text, your job is to output information that matches user's request. Reply with the answer only. Format answer as JSON list,

            """ + contentJD
                return strOut

            try:
                result = agent.run_sync(promptTemplate.value)
                print(f"{promptTemplate.name} : {result.output}")
                llmResult.dict_of_results[promptTemplate.name] = result.output
            except pydantic_ai.exceptions.UnexpectedModelBehavior:
                print(f"{promptTemplate.name} : []")
                llmResult.dict_of_results[promptTemplate.name] = OneResultList(results = [])

        # output summary in JSON alongside the original job description
        jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobDescriptionPath).stem + ".json"
        with open(jobDescriptionJSON, "w") as jsonOut:
            jsonOut.writelines(llmResult.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
