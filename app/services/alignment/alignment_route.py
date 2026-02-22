"""API routes for alignment CSV management."""
from pathlib import Path
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File

router = APIRouter()

# Singleton instance - shared across requests
_alignment_service = None


def get_alignment_service():
    """Get or create alignment service singleton."""
    global _alignment_service
    if _alignment_service is None:
        from .alignment import Alignment
        _alignment_service = Alignment()
    return _alignment_service


@router.post("/upload-alignment-csv")
async def upload_alignment_csv(file: UploadFile = File(...)):
    """
    Upload new alignment CSV and hot-reload data.
    
    This replaces the current CSV file and rebuilds all indices
    (in-memory + ChromaDB) without restarting the service.
    
    Returns:
        Stats about uploaded data including counts by type
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file type. Only CSV files are accepted."
            )
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Save to data directory
        csv_path = Path(__file__).parent / "data" / "alignments.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write new CSV
        csv_path.write_bytes(content)
        
        # Hot-reload alignment service
        alignment_service = get_alignment_service()
        stats = alignment_service.reload_data()
        
        return {
            "status": "success",
            "message": f"CSV '{file.filename}' uploaded successfully",
            "file": file.filename,
            "csv_path": str(csv_path),
            **stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload CSV: {str(e)}"
        )


@router.get("/alignment-health")
async def alignment_health():
    """
    Check if alignment data is loaded and get basic info.
    
    Returns:
        Health status and data load info
    """
    try:
        alignment_service = get_alignment_service()
        stats = alignment_service.get_stats()
        
        is_healthy = (
            stats["answers_count"] > 0 and 
            stats["alignments_count"] > 0
        )
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "data_loaded": is_healthy,
            "answers_count": stats["answers_count"],
            "alignments_count": stats["alignments_count"],
            "last_updated": stats["last_updated"],
            "csv_path": stats["csv_path"]
        }
    
    except Exception as e:
        return {
            "status": "error",
            "data_loaded": False,
            "error": str(e)
        }


@router.get("/alignment-stats")
async def alignment_stats():
    """
    Get detailed statistics about loaded alignment data.
    
    Returns:
        Detailed stats including counts by type
    """
    try:
        alignment_service = get_alignment_service()
        stats = alignment_service.get_stats()
        
        return {
            "status": "success",
            **stats
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )
