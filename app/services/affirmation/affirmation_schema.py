from pydantic import BaseModel, Field
from typing import Optional,List, Any, Dict

class HistoryItem(BaseModel):
    message: str
    response: str

class SubQuestion(BaseModel):
    sub_question: str
    sub_answers: List[str]

class QuizItem(BaseModel):
    question: str
    answers: Optional[List[str]] = None
    sub_questions: Optional[List[SubQuestion]] = None 
    
class ScentDirections(BaseModel):
    """Scent options for a single goal, split by emotional direction."""
    model_config = {"populate_by_name": True}

    Neutral: List[str]
    Elevating_Energizing: List[str] = Field(alias="Elevating/Energizing")
    Calming_Grounding: List[str] = Field(alias="Calming/Grounding")

class ScentItem(BaseModel):
    goal: str
    directions: ScentDirections
    
class affirmation_request(BaseModel):
    existing_profile_tags: Optional[List[str]] = None
    quizdata: List[QuizItem]
    synergies: Optional[Dict[str, Any]]
    harmonies: Optional[Dict[str, Any]]
    resonances: Optional[Dict[str, Any]]
    polarities: Optional[Dict[str, Any]]
    past_theme: Optional[List[str]] = None
    past_affirmations: Optional[List[List[str]]] = None
    religious_or_spritual_preference: Optional[str] = None
    religious_preference_priority_score: Optional[int] = None
    holiday_preference: Optional[str] = None
    astrology_preference: Optional[str] = None
    affirmation_type: str
    base_scent_info: List[ScentItem]
    user_id:str
    
class affirmation_response(BaseModel):
    affirmation: List[str]
    affirmation_theme: str
    short_summary_of_quiz: str
    base_scent: List[str] | str
    tertiary_scent: List[str] | str


