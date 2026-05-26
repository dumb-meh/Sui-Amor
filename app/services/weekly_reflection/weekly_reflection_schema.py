from pydantic import BaseModel
from typing import Dict, List, Any, Optional


class ReflectionRequest(BaseModel):
    previous_reflections: Optional[List[str]] = None
    goal: Optional[str] = None

class ReflectionResponse(BaseModel):
    suggestion: str
