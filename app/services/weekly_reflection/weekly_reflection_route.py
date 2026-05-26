from fastapi import APIRouter, HTTPException
from .weekly_reflection import Reflection
from .weekly_reflection_schema import ReflectionRequest, ReflectionResponse

router = APIRouter()
reflection = Reflection()


@router.post("/generate_weekly_reflection", response_model=ReflectionResponse)
async def generate_weekly_reflection(request: ReflectionRequest):
    """Generate weekly reflection and growth suggestions based on goals and previous reflections."""
    try:
        response = reflection.generate_weekly_reflection(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))