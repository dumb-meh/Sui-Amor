from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class QuizEvaluationRequest(BaseModel):
    quiz_data: List[Dict[str, Any]]

class QuizEvaluationResponse(BaseModel):
    synergies: Optional[Dict[str, Any]]
    harmonies: Optional[Dict[str, Any]]
    resonances: Optional[Dict[str, Any]]
    polarities: Optional[Dict[str, Any]]
    profile_tags: Optional[List[str]]