from fastapi import APIRouter, HTTPException
from .support_intention import SupportIntention
from .support_intention_schema import SupportIntentionRequest, SupportIntentionResponse

router = APIRouter()
support_intention = SupportIntention()


@router.post("/generate_support_intention", response_model=SupportIntentionResponse)
async def generate_support_intention(request: SupportIntentionRequest):
    """Generate support intention based on goals and previous reflections."""
    try:
        response = support_intention.generate_support_intention(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))