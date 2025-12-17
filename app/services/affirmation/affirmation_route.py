from fastapi import APIRouter, HTTPException
from .affirmation import Affirmation
from .affirmation_schema import affirmation_response, affirmation_request

router = APIRouter()
affirmation= Affirmation()     

@router.post("/daily_affirmation", response_model=affirmation_response)
async def  get_affirmation():
    try:
        response = affirmation.get_daily_affirmation()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monthly_affirmation", response_model=affirmation_response)
async def  get_affirmation():
    try:
        response = affirmation.get_monthly_affirmation()
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))