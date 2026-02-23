"""API routes for alignment CSV management."""
import io
from pathlib import Path
from typing import Dict, Any

import pandas as pd
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
    Upload new alignment CSV/Excel and hot-reload data.
    
    Accepts CSV (.csv) or Excel (.xlsx, .xls) files.
    File is always saved as 'alignments.csv' regardless of upload name.
    Rebuilds all indices (in-memory + ChromaDB) without restarting the service.
    
    Returns:
        Stats about uploaded data including counts by type
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        filename_lower = file.filename.lower()
        allowed_extensions = ('.csv', '.xlsx', '.xls')
        
        if not filename_lower.endswith(allowed_extensions):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Only CSV (.csv) or Excel (.xlsx, .xls) files are accepted. Got: {file.filename}"
            )
        
        # Read file content
        content = await file.read()
        
        if not content:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Determine if it's Excel or CSV
        is_excel = filename_lower.endswith(('.xlsx', '.xls'))
        
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataframe from uploaded content
        if is_excel:
            df = pd.read_excel(io.BytesIO(content))
            # Save as CSV (always use .csv extension)
            csv_path = data_dir / "alignments.csv"
            df.to_csv(csv_path, index=False)
        else:
            # Save CSV directly
            csv_path = data_dir / "alignments.csv"
            csv_path.write_bytes(content)
        
        # Hot-reload alignment service
        alignment_service = get_alignment_service()
        stats = alignment_service.reload_data()
        
        return {
            "status": "success",
            "message": f"File '{file.filename}' uploaded and saved as 'alignments.csv'",
            "original_file": file.filename,
            "saved_as": "alignments.csv",
            "file_type": "Excel" if is_excel else "CSV",
            "csv_path": str(csv_path),
            **stats
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to upload file: {str(e)}"
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


@router.get("/synergies")
async def get_synergies():
    """
    Get all SYNERGY alignments available in the CSV.
    
    Returns:
        List of all synergy alignments
    """
    try:
        alignment_service = get_alignment_service()
        results = alignment_service.data_store.get_by_type("SYNERGY")
        
        return {
            "type": "SYNERGY",
            "count": len(results),
            "items": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get synergies: {str(e)}"
        )


@router.get("/harmonies")
async def get_harmonies():
    """
    Get all HARMONY alignments available in the CSV.
    
    Returns:
        List of all harmony alignments
    """
    try:
        alignment_service = get_alignment_service()
        results = alignment_service.data_store.get_by_type("HARMONY")
        
        return {
            "type": "HARMONY",
            "count": len(results),
            "items": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get harmonies: {str(e)}"
        )


@router.get("/resonances")
async def get_resonances():
    """
    Get all RESONANCE alignments available in the CSV.
    
    Returns:
        List of all resonance alignments
    """
    try:
        alignment_service = get_alignment_service()
        results = alignment_service.data_store.get_by_type("RESONANCE")
        
        return {
            "type": "RESONANCE",
            "count": len(results),
            "items": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get resonances: {str(e)}"
        )


@router.get("/polarities")
async def get_polarities():
    """
    Get all POLARITY alignments available in the CSV.
    
    Returns:
        List of all polarity alignments
    """
    try:
        alignment_service = get_alignment_service()
        results = alignment_service.data_store.get_by_type("POLARITY")
        
        return {
            "type": "POLARITY",
            "count": len(results),
            "items": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get polarities: {str(e)}"
        )


@router.get("/solos")
async def get_solos():
    """
    Get all SOLO alignments available in the CSV.
    
    Returns:
        List of all solo alignments
    """
    try:
        alignment_service = get_alignment_service()
        results = alignment_service.data_store.get_by_type("SOLO")
        
        return {
            "type": "SOLO",
            "count": len(results),
            "items": results
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get solos: {str(e)}"
        )

