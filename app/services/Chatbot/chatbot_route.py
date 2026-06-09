from fastapi import APIRouter, HTTPException, Header
from .chatbot_schema import ChatbotMessageRequest, ChatbotMessageResponse
from .chatbot import ChatbotService

router = APIRouter()
chatbot_service = ChatbotService()





@router.post("/", response_model=ChatbotMessageResponse)
async def chatbot(
    request: ChatbotMessageRequest,

):

    user_id = request.user_id
    try:
        response = chatbot_service.get_response(user_id, request.user_message, "web")
        return ChatbotMessageResponse(chatbot_reply=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

