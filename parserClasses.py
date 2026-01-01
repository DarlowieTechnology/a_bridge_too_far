from typing import Optional
from typing import List
from typing import Any


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
    Identifier follows Exploitability field
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
    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
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
    
    def __hash__(self):
        return hash((self.title, self.risk, self.impact, self.exploitability, self.identifier))

    # bm25s tokenization - identifier and title
    def bm25s(self):
        return "\n".join([self.identifier, self.title])


class ReportFindingWithStatus(BaseModel):
    """A finding description in cyber security report. 
    This is a section of the report.
    Section starts with Title field.
    Risk field follows Title field.
    Impact field follows Risk field.
    Exploitability field follows Impact field.
    Identifier field follows Exploitability field.
    Status field follows Identifier field 
    Category field follows Status field
    Optional Component field follows Category field
    Optional Location follows Component field
    Optional Impact follows Location field
    Optional Description field follows Impact field
    Optional Recommendation field follows Description field.
    """

    # ^ Doc-string for the finding in the test report.
    # This doc-string is sent to the LLM as the description of the schema. 
    # This doc-string helps to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: str = Field(default=None, description="title field contains name of the issue.")
    risk: str = Field(default=None, description="risk field contains rate of the risk of the issue.")
    impact: str = Field(default=None, description="impact field contains rate of impact of the issue.")
    exploitability: str = Field(default=None, description="exploitability field contains rate of successful exploitation of the issue.")
    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
    status: str = Field(default=None, description="status field contains status of the issue.")
    category: Optional[str] = Field(default=None, description="category field follows identifier.")
    component: Optional[str] = Field(default=None, description="component field contains name of component where the issue was found.")
    location: Optional[str] = Field(default=None, description="location field contains location of the issue.")
    impactdescription: Optional[str] = Field(default=None, description="impact description field contains description of impact of the issue.")
    description: Optional[str] = Field(default=None, description="description field contains description of the issue.")
    recommendation: Optional[str] = Field(default=None, description="recommendation field contains recommendation on how to mitigate the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.title == other.title and self.risk == other.risk and self.impact == other.impact and self.exploitability == other.exploitability and self.identifier == other.identifier and self.status == other.status

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.title, self.risk, self.impact, self.exploitability, self.identifier, self.status))

    # bm25s tokenization - identifier and title
    def bm25s(self):
        return "\n".join([self.identifier, self.title])

    
class ReportIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report.
    Section starts with identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title field follows identifier field.
    Risk rating field follows title.
    Optional Status field follows Risk Rating field.
    Optional Description text section follows Status field
    Optional Recommendation text section follows Description field.
    Optional Affects field follows Recommendation field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
    title: str = Field(default=None, description="title field contains name of the issue.")
    risk: str = Field(default=None, description="risk field contains rate of the risk of the issue.")
    status: Optional[str] = Field(default=None, description="status field contains status of the issue.")
    description: Optional[str] = Field(default=None, description="description field contains description of the issue.")
    recommendation: Optional[str] = Field(default=None, description="recommendation field contains recommendation on how to mitigate the issue.")
    affects:Optional[str] = Field(default=None, description="affects field follows recommendation text section.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk))

    # bm25s tokenization - identifier and title
    def bm25s(self):
        return "\n".join([self.identifier, self.title])


class ReportIssueCD(BaseModel):
    """ReportIssueCD : An issue description in cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier field.
    Risk Rating field follows title field
    Title field repeats after Risk Rating field
    Description: field  follows second title field
    Optional Recommendation: text section follows description field
    Optional Affects: field follows Recommendation field
    Optional References: field follows Affects field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
    title: str = Field(default=None, description="title field contains name of the issue.")
    risk: str = Field(default=None, description="risk field contains rate of the risk of the issue")
    description: str = Field(default=None, description="description field contains description of the issue.")
    recommendation: Optional[str] = Field(default=None, description="recommendation field follows description field and contains recommendation on how to mitigate the issue.")
    affects:Optional[str] = Field(default=None, description="affects field follows recommendation field.")
    references:Optional[str] = Field(default=None, description="references field follows affects field.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk and self.description == other.description

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk, self.description))

    # bm25s tokenization - identifier and title
    def bm25s(self):
        return "\n".join([self.identifier, self.title])



class IssueDescription(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report. 
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier field.
    Risk rating follows Title, Risk Rating field is enclosed in brackets.
    Optional Description field follows Risk Rating field.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
    title: str = Field(default=None, description="title field contains name of the issue.")
    risk: str = Field(default=None, description="risk field contains rate of the risk of the issue.")    
    description: Optional[str] = Field(default=None, description="description field contains description of the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk))

    # bm25s tokenization - identifier and title
    def bm25s(self):
        return "\n".join([self.identifier, self.title])


class ISECIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report.
    Section starts with Title field
    Class field follows Title field 
    Severity field follows Class field 
    Difficulty field follows Severity field 
    Identifier follows Difficulty field. Finding ID contains letters, numbers, dashes, no whitespace.
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
    identifier: str = Field(default=None, description="identifier field contains unique identifier of the issue.")
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
    
    def __hash__(self):
        return hash((self.title, self.vulnClass, self.severity, self.difficulty, self.identifier))

    # bm25s tokenization - identifier and title
    def bm25s(self):
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

    def __hash__(self):
        return hash((self.identifier, self.project_key, self.project_name, self.status_category_key, 
                    self.priority_name, self.issue_updated, self.status_name,
                    self.summary, self.progress, "|".join(self.worklog)))

    # join all fields for bm25s tokenization
    def bm25s(self):
        return "\n".join([self.identifier, self.summary, self.project_key, self.project_name, self.status_category_key, 
                    self.priority_name, self.issue_updated, self.status_name,
                    self.progress, "|".join(self.worklog)])
