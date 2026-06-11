from typing import List, Optional
from pydantic import BaseModel

class HistoryItem(BaseModel):
    feeling: str
    reason: str
    ai_response: str

class ChatbotMessageRequest(BaseModel):
    user_id: str
    feeling: str
    reason:  str

class ChatbotMessageResponse(BaseModel):
    chatbot_reply: str