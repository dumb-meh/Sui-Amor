from pydantic import BaseModel
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
    
class affirmation_request(BaseModel):
    existing_profile_tags: Optional[List[str]] = None
    quizdata: List[QuizItem]
    synergies: Optional[Dict[str, Any]]
    harmonies: Optional[Dict[str, Any]]
    resonances: Optional[Dict[str, Any]]
    polarities: Optional[Dict[str, Any]]
    past_theme: Optional[List[str]] = None
    past_affirmations: Optional[List[List[str]]] = None
    
class affirmation_response(BaseModel):
    affirmation: List[str]
    affirmation_theme:str

