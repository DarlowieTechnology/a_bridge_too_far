#
# read scenarios from "scenario.json"
# skip scenarios with updated description
# pass to LLM to expand the name field into paragraph in description.
# write scenarios to "scenario.new.json"

import sys
import tomli
import json
from pathlib import Path

from typing import Union
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import pydantic_ai.exceptions

# local
from common import OneRecord, AllRecords, ConfigSingleton, OpenFile

#----------------------------------------------


def main():
    if len(sys.argv) < 3:
        print(f"Usage:\n\t{sys.argv[0]} CONFIG JSON\nExample: {sys.argv[0]} default.toml scenario")
        return

    try:
        with open(sys.argv[1], mode="rb") as fp:
            ConfigSingleton().conf = tomli.load(fp)
    except Exception as e:
        print(f"***ERROR: Cannot open config file {sys.argv[1]}, exception {e}")
        return
    
    jsonFile = sys.argv[2]

    boolResult, allScenariosOrError = OpenFile.readRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonFile)
    if (not boolResult):
        print(allScenariosOrError)
        return

    ollamaModel = OpenAIModel(
                        model_name=ConfigSingleton().conf["main_llm_name"], 
                        provider=OpenAIProvider(base_url=ConfigSingleton().conf["llm_base_url"])
                    )
    agent = Agent(ollamaModel, retries=1, output_retries=1)

    idxRecord = -1
    for oneRecord in allScenariosOrError.list_of_records:

        idxRecord += 1
        if len(oneRecord.description):
            print(f"Skipping completed record {idxRecord}")
            continue
        
        print(f"Processing record {idxRecord}")

        prompt = f"""Output expanded paragraph for text in brackets. 
        Do not format text with newlines. Output only the answer. ( {oneRecord.name} )"""

        try:
            result = agent.run_sync(prompt)
            oneRecord.description = result.output
        except pydantic_ai.exceptions.UnexpectedModelBehavior as e:
            print(f"ERROR: exception {e}")
            continue

        OpenFile.writeRecordJSON(ConfigSingleton().conf["sqlite_datapath"], jsonFile, allScenariosOrError)


if __name__ == "__main__":
    main()
