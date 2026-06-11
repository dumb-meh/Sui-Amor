from typing import Optional

from pydantic import BaseModel, Field


class QuotationItem(BaseModel):
	id: str
	goal: Optional[str] = None
	quote: str
	attribution_display: Optional[str] = Field(default=None, alias="attribution_display")
	content_type: Optional[str] = Field(default=None, alias="content_type")
	source_genre: Optional[str] = Field(default=None, alias="source_genre")
	source_work_or_reference: Optional[str] = Field(default=None, alias="source_work_or_reference")
	risk_number: Optional[int] = Field(default=None, alias="risk_number")
	filter: Optional[str] = None
	goal_tags: Optional[list[str]] = Field(default=None, alias="goal_tags")
	intensity: Optional[int] = None
	religious_filter: Optional[str] = Field(default=None, alias="religious_filter")
	action_style: Optional[str] = Field(default=None, alias="action_style")
	energy_type: Optional[str] = Field(default=None, alias="energy_type")
	notes: Optional[str] = None


class QuotationUploadResponse(BaseModel):
	total_quotes: int
	message: str


class QuotationCountResponse(BaseModel):
	total_quotes: int


class QuotationSelectionRequest(BaseModel):
	user_id: str


class QuotationSelectionResponse(BaseModel):
	quotation: QuotationItem
	history_size: int
	exhausted_cycle: bool = False
