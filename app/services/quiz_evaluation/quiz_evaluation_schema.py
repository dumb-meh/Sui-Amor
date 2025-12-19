from pydantic import BaseModel
from typing import Dict,List, Any


class QuizEvaluationRequest(BaseModel):
    quiz_data: List[Dict[str, Any]]

class QuizEvaluationResponse(BaseModel):
    synergies: Dict[str, Any]
    harmonies: Dict[str, Any]
    resonances: Dict[str, Any]
    polarities: Dict[str, Any]
    profile_tags: List[str]