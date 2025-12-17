import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.vectordb.vectordb_ingestion import AlignmentIngestionService

router = APIRouter(prefix="/alignments", tags=["Alignments"])


def get_ingestion_service() -> AlignmentIngestionService:
    return AlignmentIngestionService()


@router.post("/upload")
async def upload_alignment_file(
    file: UploadFile = File(...),
    reset_collection: bool = True,
    service: AlignmentIngestionService = Depends(get_ingestion_service),
):
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        buffer.name = file.filename or "upload.bin"
        result = service.ingest_file(buffer, reset_collection=reset_collection)
        return {"uploaded": file.filename, **result}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
