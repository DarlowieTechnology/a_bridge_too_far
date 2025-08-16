#
# read list of dict from "RECORDS.json"
# sort by name
# write "RECORDS.new.json"

import sys
import tomli
import json
from pathlib import Path

from typing import Union
from operator import itemgetter


# local
from common import ConfigSingleton
from common import OpenFile

#----------------------------------------------

def readProdCatJSON() -> tuple[bool, Union[dict, str]] : 

    dataPath = ConfigSingleton().conf["sqlite_datapath"]
    folder_path = Path(dataPath)
    if not folder_path.is_dir():
        return False, f"***ERROR: data path is not a folder: {folder_path}"
    jsonFileName = str(folder_path) + '/productfeature.json'
    boolResult, contentOrError = OpenFile.open(filePath = jsonFileName, readContent = True)
    if not boolResult:
        return False, contentOrError
    sourceJson = json.loads(contentOrError)
    allRecords = {}
    allRecords["list_of_records"] = []
    for oneDict in sourceJson["list_of_records"]:
        allRecords["list_of_records"].append(dict(**oneDict))
    return True, allRecords

def writeNewProdCatJSON(allProdCats : dict) -> bool:
    """write all updated prod cats to new prod cats file"""

    dataPath = ConfigSingleton().conf["sqlite_datapath"]
    folder_path = Path(dataPath)
    jsonFileName = str(folder_path) + '/productfeature.new.json'
    with open(jsonFileName, "w") as jsonOut:
        json.dump(allProdCats, jsonOut, indent=4)
    return True

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

    boolResult, allProdCatOrError = readProdCatJSON()
    if (not boolResult):
        print(allProdCatOrError)
        return

    allProdCatOrError["list_of_records"].sort(key=itemgetter('name'))

    writeNewProdCatJSON(allProdCatOrError)


if __name__ == "__main__":
    main()
