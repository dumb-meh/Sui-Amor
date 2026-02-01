import io
from enum import Enum

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.vectordb.vectordb_ingestion import AlignmentIngestionService

router = APIRouter(prefix="/alignments", tags=["Alignments"])


def get_ingestion_service() -> AlignmentIngestionService:
    return AlignmentIngestionService()


class ResetOption(str, Enum):
    yes = "yes"
    no = "no"


@router.post("/upload")
async def upload_alignment_file(
    file: UploadFile = File(...),
    reset_collection: ResetOption = ResetOption.yes,
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        buffer.name = file.filename or "upload.bin"
        should_reset = reset_collection == ResetOption.yes
        result = service.ingest_file(buffer, reset_collection=should_reset)
        return {"uploaded": file.filename, **result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/harmonies")
async def get_harmonies(
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        results = service.vector_store.get_by_type("harmonies")
        return {"type": "harmonies", "count": len(results), "results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/polarities")
async def get_polarities(
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        results = service.vector_store.get_by_type("polarities")
        return {"type": "polarities", "count": len(results), "results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/resonances")
async def get_resonances(
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        results = service.vector_store.get_by_type("resonances")
        return {"type": "resonances", "count": len(results), "results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/synergies")
async def get_synergies(
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        results = service.vector_store.get_by_type("synergies")
        return {"type": "synergies", "count": len(results), "results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@router.get("/query")
async def query_alignments(
    q: str,
    limit: int = 10,
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    """
    Run a vector query for `q` and return the nearest alignment records.
    """
    try:
        results = service.query_by_text(q, limit=limit)
        return {"query": q, "limit": limit, "count": len(results), "results": results}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
