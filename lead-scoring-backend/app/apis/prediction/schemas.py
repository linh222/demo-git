from pydantic import BaseModel, Field


class FeatureModel(BaseModel):
    lead_id: str = Field(..., min_length=1, max_length=32, description="lead id")
    date: str = Field(..., min_length=4, max_length=32, description="Date request")
    phone_type: str = Field(..., max_length=20, description="Type of phone")
    telco: str = Field(..., max_length=64, description="Telco")
    starting_09X: str = Field(..., max_length=6, description="YES or NO value")
    starting_08X: str = Field(..., max_length=6, description="YES or NO value")
    is_gmail: str = Field(..., max_length=6, description="YES or NO value")
    is_yahoo_email: str = Field(..., max_length=6, description="YES or NO value")
    is_educational_email: str = Field(..., max_length=6, description="YES or NO value")
    package: str = Field(..., max_length=128, description="Package name")
    entity_lead_source: str = Field(..., max_length=128, description="Entity lead source")
    os_version: str = Field(..., max_length=128, description="Os version")
    browser_type: str = Field(..., max_length=128, description="Browser type")

    class Config:
        orm_mode = True

    def __str__(self):
        return f"{self.lead_id}, {self.date}, {self.phone_type}, {self.telco}, {self.starting_09X}," \
               f"{self.starting_08X}, {self.is_gmail}, {self.is_yahoo_email}, {self.is_educational_email}," \
               f"{self.package}, {self.entity_lead_source}, {self.os_version}, {self.browser_type},"


class ResponseModel(BaseModel):
    lead_id: str = Field(..., min_length=1, max_length=32, description="lead id")
    lead_score: float = Field(..., description="Score after predict")
    date: str = Field(..., min_length=4, max_length=32, description="Date request")
    model_version: str = Field(..., max_length=20, description="Model version")

    class Config:
        orm_mode = True
