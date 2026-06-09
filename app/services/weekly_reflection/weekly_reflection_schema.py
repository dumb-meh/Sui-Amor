from pydantic import BaseModel
from typing import List, Optional


class ReflectionRequest(BaseModel):
    user_id: str


class ReflectionResponse(BaseModel):
    suggestion: str
