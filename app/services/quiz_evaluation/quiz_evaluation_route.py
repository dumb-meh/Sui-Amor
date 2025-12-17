from fastapi import APIRouter, HTTPException

from .quiz_evaluation import QuizEvaluation
from .quiz_evaluation_schema import QuizEvaluationRequest, QuizEvaluationResponse

router = APIRouter()
quiz_evaluation = QuizEvaluation()

@router.post("/quiz_evaluation", response_model=QuizEvaluationResponse)
async def get_quiz_evaluation(request: QuizEvaluationRequest):
    try:
        response = quiz_evaluation.quiz_evaluation(request)
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
