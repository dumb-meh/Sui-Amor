from fastapi import APIRouter, HTTPException, Query
from .support_intention import SupportIntention
from .support_intention_schema import SupportIntentionRequest, SupportIntentionResponse

router = APIRouter(tags=["Support Intention"])
support_intention_service = SupportIntention()


@router.get("/support_intention", response_model=SupportIntentionResponse, summary="Get the current support intention")
async def get_support_intention(user_id: str = Query(..., description="The user's unique identifier")):
    """
    Returns the current support intention for the user.

    Behaviour:
    - If no data exists yet → generate, save, return.
    - If data exists but is >= 7 days old → silently regenerate, save, return fresh content.
    - Otherwise → return the cached value instantly.

    The frontend should always call this endpoint; it never needs to call /generate directly.
    """
    try:
        cached, is_stale = support_intention_service.get_current_intention(user_id)

        if cached is None or is_stale:
            # First time or weekly refresh — generate transparently
            return support_intention_service.generate_support_intention(user_id)

        return cached
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/support_intention/generate", response_model=SupportIntentionResponse, summary="Force-generate a support intention")
async def force_generate_support_intention(request: SupportIntentionRequest):
    """
    Force-generates a fresh support intention regardless of age.
    Updates Redis and returns the new content.
    Intended for admin/backend use — the frontend uses GET instead.
    """
    try:
        return support_intention_service.generate_support_intention(request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))