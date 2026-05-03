from typing import Optional
from typing import List
from typing import Any
import hashlib


from pydantic import BaseModel, Field

class ParserClassFactory(BaseModel) :

    @staticmethod
    def factory(class_name) -> BaseModel :
        classes = {
            "ReportFinding": ReportFinding,
            "ReportFindingWithStatus" : ReportFindingWithStatus,
            "ReportIssue": ReportIssue,
            "ReportIssueCD" : ReportIssueCD,
            "IssueDescription" : IssueDescription,
            "ISECIssue" : ISECIssue,
            "JiraIssueRAG" : JiraIssueRAG
        }
        return classes[class_name]



class ReportFinding(BaseModel):
    """A finding description in cyber security report. 
    This is a section of the report. The section contains information on the finding.
    Section starts with finding title field.
    Risk field follows title field.
    Impact field follows Risk field.
    Exploitability field follows Impact field.
    Identifier follows Exploitability field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.
    Category follows Identifier field
    Optional Component field follows Category field
    Optional Location field follows Component field
    Optional Impact follows Location field
    Optional Description field follows Impact field
    Optional Recommendation field follows description field.
    """

    # ^ Doc-string for the finding in the test report.
    # This doc-string is sent to the LLM as the description of the schema. 
    # This doc-string helps to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: str = Field(default=None, description="title field contains name of the issue.")
    risk: str = Field(default=None, description="risk field contains rate of the risk of the issue")
    impact: str = Field(default=None, description="impact field contains rate of impact of the issue")
    exploitability: str = Field(default=None, description="exploitability field contains rate of successful exploitation of the issue")
    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.")
    category: Optional[str] = Field(default=None, description="category field contains category of the issue")
    component: Optional[str] = Field(default=None, description="component field contains name of component where the issue was found")
    location: Optional[str] = Field(default=None, description="location field contains location of the issue")
    impactdescription: Optional[str] = Field(default=None, description="impact description field contains description of impact of the issue.")
    description: Optional[str] = Field(default=None, description="description field contains description of the issue")
    recommendation: Optional[str] = Field(default=None, description="recommendation field contains recommendation on how to mitigate the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.title == other.title and self.risk == other.risk and self.impact == other.impact and self.exploitability == other.exploitability and self.identifier == other.identifier

    def __ne__(self, other):
        return not self.__eq__(other)

    def stringToHash(self) -> str:
        return self.title + self.risk + self.impact + self.exploitability + self.identifier

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])


class ReportFindingWithStatus(BaseModel):
    """A finding description in cyber security report. 
    This is a section of the report.
    Section starts with Title field. Title field is a string terminated by word 'risk'.
    Risk field follows Title field. Risk contains one word describing risk level: undetermined or low or medium or high or critical.
    Impact field follows risk field. Impact field contains one word describing impact level: undetermined or low or medium or high or critical.
    Exploitability field follows impact field. Exploitability field contains one word describing exploitability: undetermined or low or medium or high or critical.
    Identifier field follows exploitability field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. Identifier field is terminated by word 'status'.
    Status field follows Identifier field. Status field contains one word describing status. Status field is terminated by word 'category'.
    Optional category field follows status field
    Optional component field follows category field
    Optional location follows component field
    Optional impact follows location field
    Optional description field follows impact field
    Optional recommendation field follows description field.
    """

    # ^ Doc-string for the finding in the test report.
    # This doc-string is sent to the LLM as the description of the schema. 
    # This doc-string helps to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: str = Field(default=None, description="Title field is a string terminated by word 'risk'.")
    risk: str = Field(default=None, description="Risk field follows Title field. Risk contains one word describing risk level: undetermined or low or medium or high or critical.")
    impact: str = Field(default=None, description="Impact field follows risk field. Impact field contains one word describing impact level: undetermined or low or medium or high or critical.")
    exploitability: str = Field(default=None, description="Exploitability field follows impact field. Exploitability field contains one word describing exploitability: undetermined or low or medium or high or critical.")
    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. Identifier field is terminated by word 'status'.")
    status: str = Field(default=None, description="Status field follows Identifier field. Status field contains one word describing status. Status field is terminated by word 'category'.")
    category: Optional[str] = Field(default=None, description="Optional category field follows status field.")
    component: Optional[str] = Field(default=None, description="Optional component field contains name of component where the issue was found.")
    location: Optional[str] = Field(default=None, description="location Optional field contains location of the issue.")
    impactdescription: Optional[str] = Field(default=None, description="Optional impact description field contains description of impact of the issue.")
    description: Optional[str] = Field(default=None, description="Optional description field contains description of the issue.")
    recommendation: Optional[str] = Field(default=None, description="Optional recommendation field contains recommendation on how to mitigate the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.title == other.title and self.risk == other.risk and self.impact == other.impact and self.exploitability == other.exploitability and self.identifier == other.identifier and self.status == other.status

    def __ne__(self, other):
        return not self.__eq__(other)

    def stringToHash(self) -> str:
        return self.title + self.risk + self.impact + self.exploitability + self.identifier + self.status

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])

    
class ReportIssue(BaseModel):
    """
An issue section in cyber security report. \
Section starts with identifier field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. \
Title follows identifier field. Title is a string terminated by words: 'risk rating'. \
Risk field starts after words: 'risk rating'. Risk field contains one word describing risk level: low or medium or high or critical. \
Optional status field follows risk field. Status field contains one word describing status. \
Optional Description field follows status field. Description field terminates by word 'recommendation' or end of text. \
Optional recommendation field follows description field. Recommendation field starts after word 'recommendation'. Recommendation field terminates by word 'affects' or end of text.\
Optional affects field follows recommendation field. It starts after word 'affects'. \
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.")
    title: str = Field(default=None, description="Title follows identifier field. Title is a string terminated by words: 'risk rating'.")
    risk: str = Field(default=None, description="Risk field starts after words: 'risk rating'. Risk field contains one word describing risk level: low or medium or high or critical.")
    status: Optional[str] = Field(default=None, description="Optional status field follows risk field. Status field contains one word describing status.")
    description: Optional[str] = Field(default=None, description="Optional Description field follows status field. Description field terminates by word 'recommendation' or end of text.")
    recommendation: Optional[str] = Field(default=None, description="Optional recommendation field follows description field. Recommendation field starts after word recommendation. Recommendation field terminates by word 'affects' or end of text.")
    affects:Optional[str] = Field(default=None, description="Optional affects field follows recommendation field. It starts after word 'affects'.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def stringToHash(self) -> str:
        return self.identifier + self.title + self.risk

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])


class ReportIssueCD(BaseModel):
    """
An issue section in cyber security report. \
Section starts with identifier field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. \
Title follows identifier field. Title field is a string terminated by words risk rating. \
Risk rating field follows title field. risk rating contains one word describing risk level: low or medium or high or critical. \
Optional second title field follows risk rating field. Second title field has the same string as title field. Second title field is terminated by word description. \
Optional description field follows second title field. It terminates by word recommendation or end of text. \
Optional recommendation field follows description field. It starts after word recommendation. \
Optional affects field follows recommendation field. It starts after word affects. \
Optional references field follows Affects field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.")
    title: str = Field(default=None, description="Title follows identifier field. Title field is a string terminated by words risk rating.")
    risk: str = Field(default=None, description="Risk rating field follows title field. Risk rating field contains one word describing risk level: low or medium or high or critical.")
    title2: Optional[str] = Field(default=None, description="Optional second title field follows risk rating field. Second title field has the same string as title field. Second title field is terminated by word description.")
    description: Optional[str] = Field(default=None, description="Optional description field follows second title field. It terminstes by word recommendation or end of text.")
    recommendation: Optional[str] = Field(default=None, description="Optional recommendation field follows description field. It starts after word recommendation.")
    affects:Optional[str] = Field(default=None, description="Optional affects field follows recommendation field. It starts after word affects. ")
    references:Optional[str] = Field(default=None, description="Optional references field follows Affects field.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def stringToHash(self):
        return self.identifier + self.title + self.risk

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])


class IssueDescription(BaseModel):
    """
An issue description in cyber security report. \
Section starts with identifier field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. Identifier field is terminated by semicolon.
Title field follows identifier field. Title field is a string terminated by round bracket.
Risk rating follows title field. Risk rating field is enclosed in round brackets.
Optional description field follows risk rating field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace. Identifier field is terminated by semicolon.")
    title: str = Field(default=None, description="Title field follows identifier field. Title field is a string terminated by round bracket.")
    risk: str = Field(default=None, description="Risk rating follows title field. Risk rating field is enclosed in round brackets.")
    description: Optional[str] = Field(default=None, description="description field contains description of the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def stringToHash(self) -> str:
        return self.identifier + self.title + self.risk

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])


class ISECIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report.
    Section starts with Title field
    Class field follows Title field 
    Severity field follows Class field 
    Difficulty field follows Severity field 
    Identifier follows Difficulty field. Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.
    Optional Targets field follows Identifier field
    Optional Description field follows Targets field 
    Optional Exploit scenario field follows Description field
    Optional Short Term Solution field follows Exploit Scenario field.
    Optional Long Term Solution field follows Short Term Solution field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    title: str = Field(default=None, description="title field contains name of the issue.")
    vulnClass: str = Field(default=None, description="vulnerability class of the issue")
    severity: str = Field(default=None, description="severity of the issue")
    difficulty: str = Field(default=None, description="difficulty of exploitation of the issue")
    identifier: str = Field(default=None, description="Identifier field contains single word consisting of letters, numbers, dashes, no whitespace.")
    targets: Optional[str] = Field(default=None, description="targets field contains targets for the exploitation")
    description: Optional[str] = Field(default=None, description="description field contains description of the issue.")
    exploitScenario: Optional[str] = Field(default=None, description="exploit scenario field contains detailed description of the exploit")
    shortTermSolution: Optional[str] = Field(default=None, description="short term solution field contains description of short term mitigations")
    longTermSolution: Optional[str] = Field(default=None, description="long term solution field contains description of long term mitigations")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.title == other.title and self.vulnClass == other.vulnClass and self.severity == other.severity and self.difficulty == other.difficulty and self.identifier == other.identifier

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def stringToHash(self) -> str:
        return self.title + self.vulnClass + self.severity + self.difficulty + self.identifier

    # bm25s tokenization - identifier and title
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.title])


class JiraIssueRAG(BaseModel):
    """represents data from Jira issue relevant to RAG
    This record is not used in text comprehension, rather this is a minimized
    internal Jira issue record. Actual Jira issue record is much larger and subject 
    to change without notice. Actual records are exported from Jira using REST APIs
    List of minimized records is created before vectorization
    Note that the identifier field corresponds to Jira key field - a unique identifier
    """
    identifier: str = Field(..., description="Jira issue key") 
    project_key: str = Field(..., description="Jira project key for the issue")
    project_name: str = Field(..., description="Jira project name for the issue")
    status_category_key: str = Field(..., description="Jira status key for the issue")
    priority_name: str = Field(..., description="Jira priority name for the issue")
    issue_updated: str = Field(..., description="Datetime of issue last update")
    status_name: str = Field(..., description="Jira name of status of issue")
    summary: str = Field(..., description="Issue summary")
    progress: int = Field(..., description="Issue progress")
    worklog: list[str] = Field(..., description="List of worklogs for issue")

    def stringToHash(self) -> str:
        return self.identifier + self.project_key + self.project_name + self.status_category_key + self.priority_name + self.issue_updated + self.status_name + self.summary + self.progress + "|".join(self.worklog)

    # join all fields for bm25s tokenization
    def bm25s(self) -> str:
        return "\n".join([self.identifier, self.summary, self.project_key, self.project_name, self.status_category_key, 
                    self.priority_name, self.issue_updated, self.status_name,
                    self.progress, "|".join(self.worklog)])
