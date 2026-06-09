from fastapi import APIRouter, HTTPException, Header
from .chatbot_schema import ChatbotMessageRequest, ChatbotMessageResponse
from .chatbot import ChatbotService


router = APIRouter(tags=["Chatbot"])
chatbot_service = ChatbotService()




@router.post("/", response_model=ChatbotMessageResponse, summary="Send a message to the chatbot")
async def chatbot_web_message(
    request: ChatbotMessageRequest,
):
    user_id = request.user_id
    try:
        response = chatbot_service.get_response(user_id, request.user_message, "web")
        return ChatbotMessageResponse(chatbot_reply=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


