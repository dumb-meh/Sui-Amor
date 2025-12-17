from pydantic import BaseModel
from typing import Optional,List, Any

class affirmation_response(BaseModel):
    affirmation: str

