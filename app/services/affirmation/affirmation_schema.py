from pydantic import BaseModel
from typing import Optional,List, Any

class HistoryItem(BaseModel):
    message: str
    response: str

class affirmation_request(BaseModel):
    user_id: str
    existing_profile_tags: Optional[List[str]] = None

class affirmation_response(BaseModel):
    affirmation: str

