from fastapi import APIRouter, HTTPException, Query
from .weekly_reflection import Reflection
from .weekly_reflection_schema import ReflectionRequest, ReflectionResponse

router = APIRouter()
reflection_service = Reflection()


@router.get("/weekly_reflection", response_model=ReflectionResponse)
async def get_weekly_reflection(user_id: str = Query(..., description="The user's unique identifier")):
    """
    Returns the current weekly reflection for the user.

    Behaviour:
    - If no data exists yet → generate, save, return.
    - If data exists but is >= 7 days old → silently regenerate, save, return fresh content.
    - Otherwise → return the cached value instantly.

    The frontend should always call this endpoint; it never needs to call /generate directly.
    """
    try:
        cached, is_stale = reflection_service.get_current_reflection(user_id)

        if cached is None or is_stale:
            # First time or weekly refresh — generate transparently
            return reflection_service.generate_weekly_reflection(user_id)

        return cached
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/weekly_reflection/generate", response_model=ReflectionResponse)
async def force_generate_weekly_reflection(request: ReflectionRequest):
    """
    Force-generates a fresh weekly reflection regardless of age.
    Updates Redis and returns the new content.
    Intended for admin/backend use — the frontend uses GET instead.
    """
    try:
        return reflection_service.generate_weekly_reflection(request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))