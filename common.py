from enum import Enum, Flag, IntFlag, unique, auto

from typing import Union
from typing import List
from typing import Dict
from typing import Any

import os
import sys
import json
import logging
from logging import Logger
import inspect
import tomli
import threading

from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


# local
sys.path.append("..")
sys.path.append("../..")

@unique
class COLLECTION(Enum) :
    ISSUES = "reportissues"
    JIRA = "jiraissues"

@unique
class QUERYTYPES(IntFlag) :
    NONE = 0
    ORIGINAL = auto()
    ORIGINALCOMPRESS = auto()
    HYDE = auto()
    HYDECOMPRESS = auto()
    MULTI = auto()
    MULTICOMPRESS = auto()
    REWRITE = auto()
    REWRITECOMPRESS = auto()
    BM25SORIG = auto()
    BM25SORIGCOMPRESS = auto()
    BM25PREP = auto()
    BM25PREPCOMPRESS = auto()

@unique
class TOKENIZERTYPES(IntFlag) :
    NONE = 0
    STOPWORDSEN = auto()
    STEMMER = auto()


class RecordCollection(BaseModel):
    """
    represents all findings in a report document
    """
    report: str = Field(..., description="report document name")
    finding_dict: Dict[str, Any] = Field(default=None, description="dict of issues, key by issue identifier")
    
#    def __init__(self, report, finding_dict):
#        super().__init__()
#        self.finding_dict = finding_dict

    def __getitem__(self, key):
        """Called when obj[index] is used."""
        return self.finding_dict[key]

    def __setitem__(self, key, value):
        self.finding_dict[key] = value

    def objectCount(self):
        return len(self.finding_dict)



class OneRecord(BaseModel):
    """represents one record in structured storage"""
    id : str = Field(..., description="id of records")
    name: str = Field(..., description="name of records")
    description: str = Field(..., description="description of records")

class AllRecords(BaseModel):
    """represents collection of all records in structured storage"""
    list_of_records: List[OneRecord] = Field(..., description="list of records")


class OneQueryResult(BaseModel):
    """represents one query record in generator app"""
    id: str = Field(..., description="id of record")
    name: str = Field(..., description="name of record")
    desc: str = Field(..., description="description of record")
    query: str = Field(..., description="query used")
    distance : float = Field(..., description="distance of record")

class AllQueryResults(BaseModel):
    """represents collection of all queries in generator app"""
    list_of_queryresults: List[OneQueryResult] = Field(..., description="list of query results")

class OneDesc(BaseModel):
    """represents one project description"""
    title: str = Field(..., description="title of project")
    description: str = Field(..., description="description of project")

class AllDesc(BaseModel):
    """represents resume data"""
    ad_name : str = Field(..., description="name of the job ad file")
    exec_section: OneDesc = Field(..., description="executive section")
    project_list: list[OneDesc] = Field(..., description="list of projects")

class OneResultList(BaseModel):
    """represents one results list from LLM call"""
    results_list: list[str] = Field(..., description="list of results")


class OneEmployer(BaseModel):
    """represents one previous employer"""
    name: str = Field(..., description="employer name")
    blurb: str = Field(..., description="short description")
    start: datetime = Field(..., description="employer start date")
    end: datetime = Field(..., description="employer end date")
    numJobs: int = Field(..., description="number of jobs")


class AllEmployers(BaseModel):
    """represents all employers"""
    employer_list: list[OneEmployer] = Field(..., description="list of employers")


class OneResultWithType(BaseModel):
    """one result from RAG with expected data type name and data"""
    data: str = Field(..., description="Data from RAG document field")
    parser_typename: str = Field(..., description="Type name from RAG metadata field")
    vector_dist: float = Field(..., description="Vector distance")

    def __hash__(self):
        return hash((self.data, self.parser_typename))


class ResultWithTypeList(BaseModel):
    """represents result list from LLM call"""
    results_list: list[OneResultWithType] = Field(..., description="list of results")




class ConfigSingleton(object):
    """
    single instance class represents TOML configuration file
    configuration file is in ./ directory for CLI
    configuration file is in ../ directory for webapp
    """

    _configName = 'default.toml'
    init_lock = threading.Lock()
    _conf = {}

    def __init__(self):
        """ special method __init__ is required to load configuration once """
        with self.init_lock:
            try:
                with open(self._configName, mode="rb") as fp:
                    self._conf = tomli.load(fp)
            except Exception as e:
                try:
                    self._configName = "../" + self._configName
                    with open(self._configName, mode="rb") as fp:
                        self._conf = tomli.load(fp)
                except Exception as e:
                    print(f"***ERROR: Cannot open config file {self._configName}, exception {e}")
                    sys.exit("Program terminates")
            # read ENV
            self._conf['OLLAMA_API_KEY'] = os.environ['OLLAMA_API_KEY']
            self._conf['gemini_key'] = os.environ['gemini_key']
            self._conf['mistral_key'] = os.environ['mistral_key']
            self._conf["Jira_api_token"] = os.environ['Jira_api_token']
            self._conf["Jira_user"] = os.environ['Jira_user']
            self._conf["OPENAI_API_KEY"] = os.environ['OPENAI_API_KEY']            

    def __new__(cls):
        """ overwrite of __new__ to enforce one instance via class attribute 'instance' """
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigSingleton, cls).__new__(cls)
        return cls.instance 
    
    def __getitem__(self, key):
        """Called when obj[index] is used."""
        return self._conf[key]

    def getAbsPath(this, key):
        """return absolute path value from relative path. Compatible with Django web app"""
        return Path(str(Path(__file__).parent.resolve()) + '/' + this._conf[key]).resolve()


class DebugUtils(object):
    """utility to provide debug support"""

    @staticmethod
    def pressKey(prompt = "Press c to break:", logger : logging.Logger = None) -> bool:
        """ if DEBUG log is available -  do not break, notify in log
            if running in Django environment - do not break
            Compatible with command line and Django web app
        """
        if logger:
            filename = inspect.stack()[1].filename 
            logger.debug(f"Skipping manual breakpoint in the code called by {filename}")
            return
        if not os.getenv('DJANGO_SETTINGS_MODULE'):
            name = input(prompt)
            if name == "c":
                return True
        return False

    @staticmethod
    def logPydanticObject(objToDump, objLabel, logger : Logger = None) -> bool :
        """ if DEBUG log is available -  dump in log
            if no log - dump in stdout.
            Compatible with command line and Django web app
        """
        if logger:
            if objToDump:
                logger.debug(f"\n------{objLabel}---------\n{objToDump.model_dump_json(indent=2)}")
            else:
                logger.debug("\n------logPydanticObject : None------")
        else:
            if objToDump:
                print(f"\n------{objLabel}---------\n{objToDump.model_dump_json(indent=2)}")
            else:
                print("\n------dumpPydanticObject : None------")


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

