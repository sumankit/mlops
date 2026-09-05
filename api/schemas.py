"""Pydantic request/response schemas for the dropout-risk inference API."""
from typing import Literal
from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    # Numeric features
    age: int = Field(..., ge=14, le=25)
    Medu: int = Field(..., ge=0, le=4, description="Mother's education (0-4)")
    Fedu: int = Field(..., ge=0, le=4, description="Father's education (0-4)")
    traveltime: int = Field(..., ge=1, le=4)
    studytime: int = Field(..., ge=1, le=4)
    failures: int = Field(..., ge=0, le=3)
    famrel: int = Field(..., ge=1, le=5)
    freetime: int = Field(..., ge=1, le=5)
    goout: int = Field(..., ge=1, le=5)
    Dalc: int = Field(..., ge=1, le=5, description="Workday alcohol consumption")
    Walc: int = Field(..., ge=1, le=5, description="Weekend alcohol consumption")
    health: int = Field(..., ge=1, le=5)
    absences: int = Field(..., ge=0, le=93)
    attendance_percentage: float = Field(..., ge=0, le=100)
    avg_internal_marks: float = Field(..., ge=0, le=20)
    assignment_completion_rate: float = Field(..., ge=0, le=100)
    G1: int = Field(..., ge=0, le=20)
    G2: int = Field(..., ge=0, le=20)

    # Categorical features
    school: Literal["GP", "MS"]
    sex: Literal["F", "M"]
    address: Literal["U", "R"]
    famsize: Literal["LE3", "GT3"]
    Pstatus: Literal["T", "A"]
    Mjob: Literal["teacher", "health", "services", "at_home", "other"]
    Fjob: Literal["teacher", "health", "services", "at_home", "other"]
    guardian: Literal["mother", "father", "other"]
    schoolsup: Literal["yes", "no"]
    famsup: Literal["yes", "no"]
    paid: Literal["yes", "no"]
    activities: Literal["yes", "no"]
    nursery: Literal["yes", "no"]
    higher: Literal["yes", "no"]
    internet: Literal["yes", "no"]
    romantic: Literal["yes", "no"]
    subject: Literal["Mathematics", "Portuguese"]

    class Config:
        json_schema_extra = {
            "example": {
                "age": 17, "Medu": 2, "Fedu": 2, "traveltime": 1, "studytime": 2,
                "failures": 1, "famrel": 4, "freetime": 3, "goout": 4, "Dalc": 2,
                "Walc": 3, "health": 3, "absences": 12, "attendance_percentage": 87.1,
                "avg_internal_marks": 8.5, "assignment_completion_rate": 55.0,
                "G1": 8, "G2": 9, "school": "GP", "sex": "F", "address": "U",
                "famsize": "GT3", "Pstatus": "T", "Mjob": "services", "Fjob": "other",
                "guardian": "mother", "schoolsup": "no", "famsup": "yes", "paid": "no",
                "activities": "yes", "nursery": "yes", "higher": "yes",
                "internet": "yes", "romantic": "no", "subject": "Mathematics",
            }
        }


class PredictionResponse(BaseModel):
    risk_prediction: Literal["High Risk", "Low Risk"]
    risk_probability: float
    model_name: str
    model_version: int
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: int | None = None
