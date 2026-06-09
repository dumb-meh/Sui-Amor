from pydantic import BaseModel
from typing import List, Optional


class IntentionItem(BaseModel):
    title: str
    description: str


class SupportIntentionRequest(BaseModel):
    user_id:str

class SupportIntentionResponse(BaseModel):
    suggestion: List[IntentionItem]
