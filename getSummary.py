import requests
from pydantic import BaseModel
import sys
import json
import glob
from pathlib import Path
import re
import tomli
import sqlite3

# local
from common import ConfigSingleton

# LLM model
#
modelName = r"gemma3:1b"
#modelName = r"llama3.1:latest"


# ---------------data types


class PromptTemplate(BaseModel):
    """represents prompt template with named parameters"""
    name: str = ""
    parameters: list[str]
    value: str = ""

class LLMresult(BaseModel):
    """represents collection of results from LLM calls"""
    originFile: str = ""
    dict_of_results: dict[str, list[str]]



#
# open text file
# return content as string
# if error, return empty string
#
def openTextFile(filePath : str, readContent : bool):
    file_path = Path(filePath)
    if not file_path.is_file():
        print(f"***ERROR: Error opening file {filePath}")
        return False, ""
    try:
        with open(filePath, "r") as textFile:
            if readContent:
                return True, textFile.read()
            else:
               return True, ""
    except FileNotFoundError as e:
        print(f"***ERROR: Error opening file {filePath}, exception {e}")
    except PermissionError as e:
        print(f"***ERROR: Permission error opening file {filePath}, exception {e}")
    except Exception as e:
        print(f"***ERROR: General error opening file {filePath}, exception {e}")
    return False, ""


def getRolesFromDatabase() -> str:
    retStr = ""
    sqlDBName = ConfigSingleton().conf["database_name"]
    with sqlite3.connect(sqlDBName) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM position;")
            rows = cursor.fetchall()
            for row in rows:
                retStr = retStr + "\n" + dict(row)["name"]
        except sqlite3.OperationalError as e:
            print(f"***ERROR: Failed to access database:{e}")
        except Exception as e:
            print(f"***ERROR: General error on accessing database:{e}")
        return retStr

def getCertificationsFromDatabase() -> str:
    retStr = ""
    sqlDBName = ConfigSingleton().conf["database_name"]
    with sqlite3.connect(sqlDBName) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM certifications;")
            rows = cursor.fetchall()
            for row in rows:
                retStr = retStr + "\n" + dict(row)["name"]
        except sqlite3.OperationalError as e:
            print(f"***ERROR: Failed to access database:{e}")
        except Exception as e:
            print(f"***ERROR: General error on accessing database:{e}")
        return retStr

def fillPromptTemplate(objPrompt:PromptTemplate, jobAdText: str) -> str:
    dictParams = {}
    for param in objPrompt.parameters:
        match param:
            case "jobdescription":
                dictParams["jobdescription"] = jobAdText
            case "roles":
                dictParams["roles"] = getRolesFromDatabase()
            case "certifications":
                dictParams["certifications"] = getCertificationsFromDatabase()

    formattedPrompt = (objPrompt.value).format(**dictParams)
    return formattedPrompt

#
# read prompts from JSON file
# return list of promptTemplate objects
#
def readPrompts() -> list[PromptTemplate]:
    promptFile = ConfigSingleton().conf["prompts"]
    list_of_templates: list[PromptTemplate] = []
    boolResult, sourceStr = openTextFile(filePath = promptFile, readContent = True)
    if not boolResult:
        print("***ERROR: no prompts for LLM found")
        return []
    sourceJson = json.loads(sourceStr)
    for prompt in sourceJson:
        list_of_templates.append(PromptTemplate(
                name=prompt["name"], 
                parameters=prompt["parameters"],
                value=prompt["value"]
            ))

    return list_of_templates
    

#
# return checked list of text files
# on errors return empty list
#
def readListOfJobs() -> list[str]:
    dataPath = ConfigSingleton().conf["input_folder"]
    files: list[str] = []
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        print(f"***ERROR: data path is not a folder: {folder_path}")
        return []
    rawFiles = glob.glob(dataPath + "/*.txt")
    for file in rawFiles:
        boolResult, sourceStr = openTextFile(filePath = file, readContent = False)
        if not boolResult:
            return []
        files.append(file)
    return files


def callAPIOnce(prompt : str):
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            "model": modelName,
            "prompt" : prompt,
            "stream": False,
            "format": {
                "type": "array"
            }
        }
    )
    if response.status_code == 200:
        return response.json()['response']
    else:
        print(f"ERROR: CODE {response.status_code} DESCRIPTION {response.text}")
        return ""

#
# 
#
def callAPIs(jobDescription:str, list_of_templates:list[PromptTemplate]):
    dict_of_results: dict[str, list[str]] = {}
    for promptTemplate in list_of_templates:
        formattedPrompt = fillPromptTemplate(promptTemplate, jobDescription)
        response = callAPIOnce(formattedPrompt)

        print(response)

        # you might think API returns JSON, but it is not
        #jsonObj = json.loads(response)

        # need manual transform
        res2 = response.translate(str.maketrans({'"':' ', '[':' ', ']':' '}))
        res = re.split(r'[,]+', res2)
        values = []
        for val in res:
            values.append(val.strip())
        dict_of_results[promptTemplate.name] = values
    return dict_of_results



# Seek adds question in known fixed format after this tag:
# ^Employer questions$
# ^Your application will include the following questions:$
#
def processQuestions(jobDescription:str) :
    pattern = r'^Employer questions.*\nYour application will include the following questions:.*\n'
    matchQuestion = re.search(pattern, jobDescription, re.MULTILINE)
    if matchQuestion:
        questionsStr = jobDescription[matchQuestion.end(0):]
        return questionsStr.splitlines(False)
    return []


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
    
    listFilePaths = readListOfJobs()
    if (len(listFilePaths) == 0):
        print(f"***ERROR: no input files in {ConfigSingleton().conf["input_folder"]}")
        return
    print(f"Found {len(listFilePaths)} input files")

    templates = readPrompts()

    for jobDescriptionPath in listFilePaths:
        print(f"Processing {jobDescriptionPath}")
        llmResult = LLMresult(originFile = jobDescriptionPath, dict_of_results = {})
        boolResult, contentJD = openTextFile(filePath = jobDescriptionPath, readContent = True)
        if not boolResult:
            continue
        llmResult.dict_of_results = callAPIs(jobDescription = contentJD, list_of_templates = templates)
        llmResult.dict_of_results["questions"] = processQuestions(jobDescription = contentJD)
        
        # output summary in JSON alongside the original job description
        jobDescriptionJSON = ConfigSingleton().conf["input_folder"] + '/' + Path(jobDescriptionPath).stem + ".json"
        with open(jobDescriptionJSON, "w") as jsonOut:
            jsonOut.writelines(llmResult.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
