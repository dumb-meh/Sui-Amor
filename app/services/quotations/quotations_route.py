from fastapi import APIRouter, File, HTTPException, UploadFile

from app.utils.cache_manager import cache_manager

from .quotations import QuotationsService
from .quotations_schema import (
	QuotationItem,
	QuotationCountResponse,
	QuotationSelectionRequest,
	QuotationSelectionResponse,
	QuotationUploadResponse,
)


router = APIRouter(prefix="/quotations", tags=["Quotations"])
service = QuotationsService()


@router.post("/upload", response_model=QuotationUploadResponse)
async def upload_quotations(file: UploadFile = File(...)):
	try:
		content = await file.read()
		total_quotes = service.upload_excel(content)
		return QuotationUploadResponse(total_quotes=total_quotes, message="Quotations uploaded successfully")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=str(exc))


@router.get("", response_model=list[QuotationItem])
async def get_all_quotations():
	try:
		return service.get_all_quotes()
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))


@router.get("/count", response_model=QuotationCountResponse)
async def get_quotations_count():
	try:
		return QuotationCountResponse(total_quotes=service.get_total_quotes())
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))


@router.post("/next", response_model=QuotationSelectionResponse)
async def get_next_quotation(request: QuotationSelectionRequest):
	try:
		goal_data = cache_manager.get_user_goal(request.user_id) or {}
		goal = goal_data.get("goal")
		religious_preference = goal_data.get("religious_preference")
		quotation, history_size, exhausted_cycle = service.get_next_quote(request.user_id, goal, religious_preference)
		return QuotationSelectionResponse(
			quotation=quotation,
			history_size=history_size,
			exhausted_cycle=exhausted_cycle,
		)
	except Exception as exc:
		raise HTTPException(status_code=404, detail=str(exc))
