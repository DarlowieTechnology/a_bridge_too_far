from typing import Union
from typing import List
from typing import Dict

import os
import json
import logging
import inspect
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field


class OneRecord(BaseModel):
    """represents one record in structured storage"""
    id : str = Field(..., description="id of records")
    name: str = Field(..., description="name of records")
    description: str = Field(..., description="description of records")

class AllRecords(BaseModel):
    """represents collection of all records in structured storage"""
    list_of_records: List[OneRecord] = Field(..., description="list of records")


class OneQueryResult(BaseModel):
    """represents one query record"""
    id: str = Field(..., description="id of record")
    name: str = Field(..., description="name of record")
    desc: str = Field(..., description="name of record")
    query: str = Field(..., description="query used")
    distance : float = Field(..., description="distance of record")

class AllQueryResults(BaseModel):
    """represents collection of all queries"""
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


class ReportIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier.
    Risk rating field follows title.
    Status field follows risk rating field
    Description text section follows status field
    Recommendation text section follows description.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

#    identifier: str = Field(default=None, pattern=r"^SR-\d\d\d-\d\d\d$", description="identifier contains letters, numbers, dashes, no whitespace")
    identifier: str = Field(default=None, description="identifier contains letters, numbers, dashes, no whitespace")
    title: str = Field(default=None, description="title field follows identifier")
    risk: str = Field(default=None, description="risk rating field follows title")    
    status: str = Field(default=None, description="status field follows risk rating ")    
    description: str = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: str = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")
    affects:str = Field(default=None, description="affects field follows recommendation text section.")

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.identifier == other.identifier and self.title == other.title and self.risk == other.risk and self.status == other.status and self.description == other.description and self.affects == other.affects

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk, self.status, self.description, self.recommendation, self.affects))    

class AllReportIssues(BaseModel):
    """
    represents all issues in a report document
    """
    name : str = Field(default=None, description="name of the report")
    pattern : str = Field(default=r"SR-\d\d\d-\d\d\d", description="pattern of separation between issues")
    issue_dict: Dict[str, ReportIssue] = Field(default=None, description="dict of issues, key by issue identifier")



class ConfigSingleton(object):
    """single instance class represents TOML configuration file"""
    conf : dict[str, str] = {}
    def __new__(cls):
        """overwrite of __new__ to enforce one instance via class attribute 'instance'"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConfigSingleton, cls).__new__(cls)
        return cls.instance 
    def getAbsPath(this, key):
        """return absolute path value from relative path. Compatible with Django web app"""
        return Path(str(Path(__file__).parent.resolve()) + '/' + this.conf[key]).resolve()


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
    def logPydanticObject(objToDump, objLabel, logger : logging.Logger = None) -> bool :
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

