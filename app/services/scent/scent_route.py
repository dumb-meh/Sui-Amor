from fastapi import APIRouter, HTTPException
from .scent import ScentGenerator
from .scent_schema import affirmation_response, affirmation_request

router = APIRouter()
affirmation = Affirmation()


@router.post("/generate_affirmations", response_model=affirmation_response)
async def generate_affirmations(request: affirmation_request):
    """Generate 12 affirmations based on quiz data and alignments."""
    try:
        response = affirmation.generate_affirmations(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))