from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class SupportIntentionRequest(BaseModel):
    previous_support_intentions: Optional[List[str]] = None
    goal: Optional[str] = None

class SupportIntentionResponse(BaseModel):
    suggestion: str
