from pydantic import BaseModel
from typing import Optional,List, Any, Dict

class HistoryItem(BaseModel):
    message: str
    response: str

class SubQuestion(BaseModel):
    sub_question: str
    sub_answers: List[str]
    
class ScentItem(BaseModel):
    goal: str
    value: List[str] | str


class QuizItem(BaseModel):
    question: str
    answers: Optional[List[str]] = None
    sub_questions: Optional[List[SubQuestion]] = None 
    
class affirmation_request(BaseModel):
    quizdata: List[QuizItem]
    base_scent_info:List[ScentItem]

    
class affirmation_response(BaseModel):
    affirmation: List[str]
    affirmation_theme:str

