from typing import Annotated
from typing import Optional

import re

from pydantic import BaseModel, Field, field_validator

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

    identifier: str = Field(default=None, description="identifier contains letters, numbers, dashes, no whitespace")
    title: str = Field(default=None, description="title field follows identifier")
    risk: str = Field(default=None, description="risk rating field follows title")    
    status: str = Field(default=None, description="status field follows risk rating ")    
    description: str = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: str = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")
    affects:str = Field(default=None, description="affects field follows recommendation text section.")

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.identifier == other.identifier and self.title == other.title and self.risk == other.risk and self.status == other.status and self.description == other.description and  self.recommendation == other.recommendation and self.affects == other.affects

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.identifier, self.title, self.risk, self.status, self.description, self.recommendation, self.affects))

    @field_validator('identifier')
    @classmethod
    def validate_id_pattern(cls, v : str):
        compiledPattern = re.compile(r"SR-\d\d\d-\d\d\d")
        match = compiledPattern.match(v) 
        if not match:
            raise ValueError('unexpected identifier format')
        return v
   

class ReportFinding(BaseModel):
    """A finding description in cyber security report. 
    This is a section of the report. The section contains information on the finding.
    Section starts with finding title field.
    Risk field follows title.
    Impact field follows Risk.
    Exploitability field follows Impact.
    Identifier follows Exploitability
    Category follows Identifier
    Component follows Category
    Location follows Component
    Impact follows Location
    Description text section follows Impact
    Recommendation text section follows description.
    """

    # ^ Doc-string for the finding in the test report.
    # This doc-string is sent to the LLM as the description of the schema Vulnerability,
    # and it can help to improve extraction results.

    # Note that:
    # Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    title: str = Field(default=None, description="finding title field")
    risk: str = Field(default=None, description="risk field follows finding title")    
    impact: str = Field(default=None, description="impact field follows risk")    
    exploitability: str = Field(default=None, description="exploitability field follows impact")    
    identifier: str = Field(default=None, description="identifier field follows exploitability")
    category: str = Field(default=None, description="category field follows identifier")
    component: str = Field(default=None, description="component field follows category")
    location: str = Field(default=None, description="location field follows component")
    impact: str = Field(default=None, description="impact field follows location")
    description: str = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: str = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.title == other.title and self.risk == other.risk and self.impact == other.impact and self.exploitability == other.exploitability and self.identifier == other.identifier and self.category == other.category and self.component == other.component and self.location == other.location and self.impact == other.impact and self.description == other.description and self.recommendation == other.recommendation

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash((self.title, self.risk, self.impact, self.exploitability, self.identifier, self.category, self.component, self.location, self.impact, self.description, self.recommendation))

    @field_validator('identifier')
    @classmethod
    def validate_id_pattern(cls, v : str):
        compiledPattern = re.compile(r"NCC-\S+-\d\d\d")
        match = compiledPattern.match(v) 
        if not match:
            raise ValueError('unexpected identifier format')
        return v
    
class CDReportIssue(BaseModel):
    """An issue description in CD cyber security report. 
    This is a section of the report. The section contains information on the issue.
    Section starts with issue identifier. Identifier contains letters, numbers, dashes, no whitespace.
    Title follows identifier.
    Risk rating field follows title.
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

    identifier: str = Field(default=None, description="identifier contains letters, numbers, dashes, no whitespace")
    title: str = Field(default=None, description="title field follows identifier")
    risk: str = Field(default=None, description="risk rating field follows title")    
    description: str = Field(default=None, description="description text section follows status and contains detailed description of the issue")
    recommendation: str = Field(default=None, description="recommendation text section follows description and contains recommendation on how to mitigate the issue.")
    affects:Optional[str] = Field(default=None, description="affects field follows recommendation text section.")

    def __eq__(self, other):
        if self.affects:
            return isinstance(other, self.__class__) and self.identifier == other.identifier and self.title == other.title and self.risk == other.risk and self.description == other.description and  self.recommendation == other.recommendation and self.affects == other.affects
        else:
            return isinstance(other, self.__class__) and self.identifier == other.identifier and self.title == other.title and self.risk == other.risk and self.description == other.description and  self.recommendation == other.recommendation

    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        if self.affects:
            return hash((self.identifier, self.title, self.risk, self.description, self.recommendation, self.affects))
        else:
            return hash((self.identifier, self.title, self.risk, self.description, self.recommendation))

    @field_validator('identifier')
    @classmethod
    def validate_id_pattern(cls, v : str):
        compiledPattern = re.compile(r"SR-\d\d\d-\d{1,2}-\d{1,2}")
        match = compiledPattern.match(v) 
        if not match:
            raise ValueError('unexpected identifier format')
        return v

