#
# Export Jira tickets to vector database
#
import sys
import logging
from pathlib import Path


from pydantic import BaseModel

from jira import JIRA

# local
import darlowie
from common import RecordCollection, ConfigCollection, OpenFile
from indexer_workflow import IndexerWorkflow
from parserClasses import ParserClassFactory


def jiraExportPhase(
        INDEXEjira_url : str, 
        INDEXEjira_max_results : int, 
        jira_user : str, 
        jira_api_token : str, 
        inputFileName : str, 
        finalJSON: str, 
        ClassTemplate : BaseModel) -> RecordCollection :
    """
    Export issues from Jira project
    Transform to smaller records
    Write as final JSON for vectorization

    Args:
        ClassTemplate (BaseModel) - issue template

    Returns:
        RecordCollection - all items
    """

    # Connect to Jira
    try:
        jira = JIRA(server=INDEXEjira_url, basic_auth=(jira_user, jira_api_token))
    except Exception as e:
        print(f"Jira API exception: {e}")
        return 0
    if not jira:
        print(f"Jira REST API connection error")
        return 0

    jql_query = f'project = {inputFileName}'
    recordCollection = RecordCollection(
        report = str(Path(inputFileName).name),
        finding_dict = {}
    )

    # Fetch issues from Jira
    # default maxResults is 50, we need more than that
    issues = jira.search_issues(jql_query, maxResults=INDEXEjira_max_results, json_result = True)
    for val in issues["issues"]:
        issueTemplate = ClassTemplate(
            identifier = val["key"],
            project_key = val["fields"]["project"]["key"],
            project_name = val["fields"]["project"]["name"],
            status_category_key = val["fields"]["statusCategory"]["key"],
            priority_name = val["fields"]["priority"]["name"],
            issue_updated = val["fields"]["updated"],
            status_name = val["fields"]["status"]["name"],
            summary = val["fields"]["summary"],
            progress = val["fields"]["progress"]["progress"],
            worklog = val["fields"]["worklog"]["worklogs"]
        )
        recordCollection.finding_dict[val["key"]] = issueTemplate

    with open(finalJSON, "w", encoding='utf8', errors='ignore') as jsonOut:
        jsonOut.writelines(recordCollection.model_dump_json(indent=2))
    return recordCollection


def main():
    context = darlowie.context

    context["statusFileName"] = context["JEXCLIstatus_FileName"]
    context["session_key"] = context["JEXCLIsession_key"]

    logging.basicConfig(stream=sys.stdout, level=logging.WARN)
    logger = logging.getLogger(context["JEXCLIsession_key"])

    context["inputFileName"] = context["GLOBALdataFolder"] + context["JEXCLIdataFolder"] + "SCRUM"
    context["rawTextFromDoc"] = context["inputFileName"] + ".raw.txt"
    context["rawJSON"] = context["inputFileName"] + ".raw.json"
    context["finalJSON"] = context["inputFileName"] + ".json"
    context["inputFileBaseName"] = "jira:SCRUM"

    configCollection = ConfigCollection(context)
    indexerWorkflow = IndexerWorkflow()
    indexerWorkflow.configure(configCollection)

    issueTemplate = ParserClassFactory.factory("JiraIssueRAG")

    exportedIssues = jiraExportPhase(
        INDEXEjira_url = configCollection["INDEXEjira_url"],
        INDEXEjira_max_results = configCollection["INDEXEjira_max_results"],
        jira_user = configCollection["jira_user"],
        jira_api_token = configCollection["jira_api_token"],
        inputFileName = configCollection["inputFileName"],
        finalJSON = configCollection["finalJSON"],
        ClassTemplate = issueTemplate
    )

    print(f"Exported {exportedIssues} Jira issues.")
    indexerWorkflow.vectorizeFinalJSONPhase(issueTemplate)



if __name__ == "__main__":
    main()
