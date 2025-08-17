from typing import Union
from typing import List

import json
from pathlib import Path
from pydantic import BaseModel, Field


class OneRecord(BaseModel):
    """represents one record in structured storage"""
    name: str = Field(..., description="name of records")
    description: str = Field(..., description="description of records")

class AllRecords(BaseModel):
    """represents collection of all records in structured storage"""
    list_of_records: List[OneRecord] = Field(..., description="list of records")


class ConfigSingleton(object):
    """single instance class represents TOML configuration file"""
    conf : dict[str, str] = {}
    def __new__(cls):
        """overwrite of __new__ to enforce one instance via class attribute 'instance'"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigSingleton, cls).__new__(cls)
        return cls.instance 

class OpenFile():

    @staticmethod
    def readListOfFileNames(inputFolder: str, globPattern : str) -> tuple[bool, Union[list[str],str]] :
        """return list of files in a folder"""
        folder_path = Path(inputFolder)
        if not folder_path.is_dir():
            return False, f"***ERROR: data path is not a folder: {folder_path}"
        fileNames = list(folder_path.glob(globPattern))
        for file in fileNames:
            boolResult, sourceStr = OpenFile.open(filePath = file, readContent = False)
            if not boolResult:
                return False, [sourceStr]
        return True, fileNames

    @staticmethod
    def open(filePath : str, readContent : bool) -> tuple[bool, str] :
        file_path = Path(filePath)
        if not file_path.is_file():
            return False, f"***ERROR: Error opening file {filePath}"
        try:
            # ignore invalid characters in text
            with open(filePath, "r", encoding='utf-8', errors='ignore') as textFile:
                if readContent:
                    return True, textFile.read()
                else:
                    return True, ""
        except FileNotFoundError as e:
            return False, f"***ERROR: Error opening file {filePath}, exception {e}"
        except PermissionError as e:
            return False, f"***ERROR: Permission error opening file {filePath}, exception {e}"
        except UnicodeDecodeError as e:
            return False, f"***ERROR: Unicode decode error in file {filePath}, exception {e}"
        except Exception as e:
            return False, f"***ERROR: General error opening file {filePath}, exception {e}"
        return False, ""

    @staticmethod
    def readRecordJSON(dataFolder: str, jsonName : str) -> tuple[bool, Union[AllRecords, str]] : 
        """read all records from JSON file. return AllRecords """

        folder_path = Path(dataFolder)
        if not folder_path.is_dir():
            return False, f"***ERROR: data path is not a folder: {folder_path}"
        jsonFileName = str(folder_path) + '/' + jsonName + '.json'
        boolResult, contentOrError = OpenFile.open(filePath = jsonFileName, readContent = True)
        if not boolResult:
            return False, contentOrError
        sourceJson = json.loads(contentOrError)
        allRecords = AllRecords(list_of_records = [])
        for oneDict in sourceJson["list_of_records"]:
            allRecords.list_of_records.append(OneRecord(**oneDict))
        return True, allRecords

    @staticmethod
    def writeRecordJSON(dataFolder: str, jsonName : str, allRecords : AllRecords) -> bool:
        """write all records to new JSON file"""

        folder_path = Path(dataFolder)
        jsonFileName = str(folder_path) + '/' + jsonName + '.new.json'
        with open(jsonFileName, "w") as jsonOut:
            jsonOut.writelines(allRecords.model_dump_json(indent=2))
        return True

