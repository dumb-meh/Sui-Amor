from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union


class SubQuestion(BaseModel):
    sub_question: str
    sub_answers: List[str]


class QuizItem(BaseModel):
    question: str
    answers: Optional[List[str]] = None
    sub_questions: Optional[List[SubQuestion]] = None


class QuizEvaluationRequest(BaseModel):
    answers: List[QuizItem]
    religious_or_spritual_preference: Optional[str] = None
    religious_or_spritual_preference: Optional[str] = None
    holiday_preference: Optional[str] = None
    astrology_preference: Optional[str] = None

class QuizEvaluationResponse(BaseModel):
    synergies: Optional[Dict[str, Any]]
    harmonies: Optional[Dict[str, Any]]
    resonances: Optional[Dict[str, Any]]
    polarities: Optional[Dict[str, Any]]
    profile_tags: Optional[List[str]]