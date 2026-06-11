from fastapi import APIRouter, HTTPException
from .daily_feelings_schema import ChatbotMessageRequest, ChatbotMessageResponse
from .daily_feelings import DailyFeelingsService


router = APIRouter(tags=["Daily Feelings"])
daily_feelings_service = DailyFeelingsService()




@router.post("/daily_feelings", response_model=ChatbotMessageResponse, summary="Send a daily feelings reflection")
async def chatbot_web_message(
    request: ChatbotMessageRequest,
):
    user_id = request.user_id
    try:
        response = daily_feelings_service.get_response(user_id, request.feeling, request.reason, "web")
        return ChatbotMessageResponse(chatbot_reply=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


