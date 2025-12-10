from typing import Optional
from typing import List
from typing import Any


from pydantic import BaseModel, Field

class ParserClassFactory(BaseModel) :

    @staticmethod
    def factory(class_name) -> BaseModel :
        classes = {
            "ReportFinding": ReportFinding,
            "ReportIssue": ReportIssue,
            "IssueDescription" : IssueDescription,
            "ISECIssue" : ISECIssue,
            "JiraIssueRAG" : JiraIssueRAG
        }
        return classes[class_name]



class ReportFinding(BaseModel):
    """A finding description in cyber security report. 
    This is a section of the report. The section contains information on the finding.
    Section starts with finding title field.
    Risk field follows title.
    Impact field follows Risk.
    Exploitability field follows Impact.
    Identifier follows Exploitability
    Category follows Identifier
    Optional Component follows Category
    Location follows Component
    Impact follows Location
    Description text section follows Impact
    Recommendation text section follows description.
    """

    # ^ Doc-string for the finding in the test report.
    # This doc-string is sent to the LLM as the description of the schema. 
    # This doc-string helps to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: str = Field(default=None, description="First field is a title field.")
    risk: str = Field(default=None, description="risk field follows finding title")    
    impact: str = Field(default=None, description="impact field follows risk")    
    exploitability: str = Field(default=None, description="exploitability field follows impact")    
    identifier: str = Field(default=None, description="The unique identifier of the issue. The identifier field follows exploitability")
    category: Optional[str] = Field(default=None, description="category field follows identifier")
    component: Optional[str] = Field(default=None, description="optional component field follows category")
    location: Optional[str] = Field(default=None, description="location field follows optional component")
    impactdescription: Optional[str] = Field(default=None, description="impact description field follows location")
    description: Optional[str] = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: Optional[str] = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.title == other.title and self.risk == other.risk and self.impact == other.impact and self.exploitability == other.exploitability and self.identifier == other.identifier

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.title, self.risk, self.impact, self.exploitability, self.identifier))

    # join attribute values for bm25s tokenization
    def bm25s(self):
        return " ".join([self.title, "\n", self.risk, self.impact, self.exploitability, self.identifier])


    
class ReportIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier.
    Risk rating field follows title.
    Status field follows Risk rating
    Description text section follows status field
    Recommendation text section follows description.
    Optional Affects field follows Recommendation
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="The unique identifier of the issue")
    title: str = Field(default=None, description="title field follows identifier")
    risk: str = Field(default=None, description="risk rating field follows title")
    status: Optional[str] = Field(default=None, description="status field follows risk rating ")
    description: Optional[str] = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: Optional[str] = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")
    affects:Optional[str] = Field(default=None, description="affects field follows recommendation text section.")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented        
        return self.identifier == other.identifier and self.title == other.title and self.risk == other.risk

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk))

    # join all fields for bm25s tokenization
    def bm25s(self):
        return " ".join([self.title, "\n", self.identifier, self.risk])


class IssueDescription(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier on the same line.
    Description text section follows title
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    identifier: str = Field(default=None, description="The unique identifier of the issue")
    title: str = Field(default=None, description="title field follows identifier on the same line")
    description: Optional[str] = Field(default=None, description="description text section follows title and contains detailed description of the issue")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.identifier == other.identifier and self.title == other.title

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title))

    # join all fields for bm25s tokenization
    def bm25s(self):
        return " ".join([self.title, "\n", self.identifier])


class ISECIssue(BaseModel):
    """An issue description in cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with title 
    Class field follows title on the new line.
    Severity field follows class field on the same line.
    Difficulty field follows severity field on the same line.
    Identifier follows Difficulty on the new line. Finding ID contains letters, numbers, dashes, no whitespace.
    Targets field follows identifier on the new line.
    Description text section follows targets field on the new line.
    Exploit scenario text section follows description text section  on the new line.
    Short Term Solution text section follows exploit scenario on the new line.
    Long Term Solution text section follows Short Term Solution on the new line.
    """

    # ^ Doc-string for the issue in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.

    title: str = Field(default=None, description="title of the issue")
    vulnClass: str = Field(default=None, description="vulnerability class of the issue")
    severity: str = Field(default=None, description="severity of the issue")
    difficulty: str = Field(default=None, description="difficulty of exploitation of the issue")
    identifier: str = Field(default=None, description="The unique identifier of the issue")
    targets: Optional[str] = Field(default=None, description="targets for the exploitation")
    description: Optional[str] = Field(default=None, description="description text section follows targets field and contains detailed description of the issue")
    exploitScenario: Optional[str] = Field(default=None, description="exploit scenario text section follows description text section and contains detailed description of the exploit")
    shortTermSolution: Optional[str] = Field(default=None, description="short term solution text section follows exploit scenario text section and contains description of short term mitigations")
    longTermSolution: Optional[str] = Field(default=None, description="long term solution text section follows short term solution text section and contains description of long term mitigations")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.title == other.title and self.vulnClass == other.vulnClass and self.severity == other.severity and self.difficulty == other.difficulty and self.identifier == other.identifier

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.title, self.vulnClass, self.severity, self.difficulty, self.identifier))

    # join all fields for bm25s tokenization
    def bm25s(self):
        return " ".join([self.title, "\n", self.vulnClass, self.severity, self.difficulty, self.identifier])



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
        return " ".join([self.identifier, "\n", self.project_key, self.project_name, self.status_category_key, 
                    self.priority_name, self.issue_updated, self.status_name,
                    self.summary, self.progress, "|".join(self.worklog)])
