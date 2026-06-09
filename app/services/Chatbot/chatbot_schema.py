from typing import List, Optional
from pydantic import BaseModel

class HistoryItem(BaseModel):
    message: str
    response: str

class ChatbotMessageRequest(BaseModel):
    user_id: str
    user_message: str

class ChatbotMessageResponse(BaseModel):
    chatbot_reply: str