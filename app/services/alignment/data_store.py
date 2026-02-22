"""Data store for alignment answers and alignments with CSV parsing and indexing."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd


class AlignmentDataStore:
    """Manages answer options and alignments from CSV with in-memory + vector indices."""
    
    def __init__(self, csv_path: Optional[str] = None, persist_dir: Optional[str] = None):
        """
        Initialize data store.
        
        Args:
            csv_path: Path to CSV file. Defaults to data/alignments.csv
            persist_dir: ChromaDB persistence directory. Defaults to .chroma
        """
        self.csv_path = Path(csv_path) if csv_path else Path(__file__).parent / "data" / "alignments.csv"
        self.persist_dir = Path(persist_dir) if persist_dir else Path.cwd() / ".chroma"
        
        # In-memory stores
        self.answers: Dict[str, Dict[str, Any]] = {}  # answer_id -> answer data
        self.alignments: Dict[str, Dict[str, Any]] = {}  # alignment_id -> alignment data
        self.last_updated: Optional[datetime] = None
        
        # ChromaDB for vector fallback (Tier 3)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None
        
        # Load data if CSV exists
        if self.csv_path.exists():
            print(f"[INFO] Loading CSV from: {self.csv_path}")
            self.reload_from_csv(self.csv_path)
        else:
            print(f"[WARNING] CSV file not found at: {self.csv_path}")
            print(f"[INFO] Service will start with empty data. Upload CSV via /upload-alignment-csv endpoint")
    
    def reload_from_csv(self, csv_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Parse CSV/Excel and rebuild all indices.
        
        Args:
            csv_path: Optional new CSV/Excel path. If None, uses existing path.
            
        Returns:
            Stats about loaded data
        """
        if csv_path:
            self.csv_path = Path(csv_path)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"File not found: {self.csv_path}")
        
        # Parse file (supports CSV and Excel)
        df = self._load_dataframe(self.csv_path)
        
        # Debug: Check what values Is_Selectable has
        if "Is_Selectable" in df.columns:
            unique_values = df["Is_Selectable"].unique()
            print(f"[DEBUG] Is_Selectable unique values: {unique_values}")
        
        # Separate answers from alignments
        # Handle both "TRUE" (string) and True (boolean)
        answers_df = df[
            (df["Is_Selectable"].astype(str).str.upper() == "TRUE") | 
            (df["Is_Selectable"] == True)
        ].copy()
        
        # Alignments have non-empty Alignment_Type
        alignments_df = df[df["Alignment_Type"].notna() & (df["Alignment_Type"].astype(str).str.strip() != "")].copy()
        
        print(f"[DEBUG] Parsed CSV: {len(answers_df)} answer rows, {len(alignments_df)} alignment rows")
        
        # Build answer dictionary
        self.answers = {}
        for _, row in answers_df.iterrows():
            answer_id = str(row["Answer_ID"]).strip()
            if not answer_id or answer_id == "nan":
                continue
            
            self.answers[answer_id] = {
                "answer_id": answer_id,
                "question_id": str(row.get("Question_ID", "")).strip(),
                "question_text": str(row.get("Question_Text", "")).strip(),
                "text": str(row.get("Answer_Text", "")).strip(),
                "category": str(row.get("Category", "")).strip(),
                "parent": str(row.get("Parent_Answer_ID", "")).strip() if pd.notna(row.get("Parent_Answer_ID")) else None,
                "axes": {
                    "energy": self._safe_float(row.get("Axis_Energy")),
                    "pace": self._safe_float(row.get("Axis_Pace")),
                    "orientation": self._safe_float(row.get("Axis_Orientation")),
                    "structure": self._safe_float(row.get("Axis_Structure")),
                    "expression": self._safe_float(row.get("Axis_Expression")),
                }
            }
        
        # Build alignment dictionary
        self.alignments = {}
        for _, row in alignments_df.iterrows():
            alignment_id = str(row["Alignment_ID"]).strip()
            if not alignment_id or alignment_id == "nan":
                continue
            
            # Parse components (e.g., "COLOR_RED+COLOR_YELLOW" -> ["COLOR_RED", "COLOR_YELLOW"])
            components_str = str(row.get("Alignment_Components", "")).strip()
            components = [c.strip() for c in components_str.split("+") if c.strip()]
            
            # Compute alignment axes as weighted average of component axes
            alignment_axes = self._compute_alignment_axes(components)
            
            # Extract categories from components
            categories = list(set([
                self.answers[comp]["category"] 
                for comp in components 
                if comp in self.answers
            ]))
            
            self.alignments[alignment_id] = {
                "id": alignment_id,
                "type": str(row.get("Alignment_Type", "")).strip().upper(),
                "name": str(row.get("Alignment_Name", "")).strip(),
                "title": str(row.get("Alignment_Name", "")).strip(),  # Alias for compatibility
                "description": str(row.get("Alignment_Text", "")).strip(),
                "components": components,
                "component_order_matters": True,  # Always true for SYNERGY/HARMONY
                "axes": alignment_axes,
                "categories": categories,
            }
        
        # Rebuild vector index for Tier 3 fallback
        print(f"[DEBUG] Starting vector index rebuild...")
        self._rebuild_vector_index()
        print(f"[DEBUG] Vector index rebuild complete")
        
        self.last_updated = datetime.now()
        
        return {
            "answers_count": len(self.answers),
            "alignments_count": len(self.alignments),
            "updated_at": self.last_updated.isoformat()
        }
    
    def _compute_alignment_axes(self, components: List[str]) -> Dict[str, float]:
        """
        Compute alignment's axis profile as weighted average of component answers.
        First component gets weight 1.0, second gets 0.5, third gets 0.33, etc.
        """
        if not components:
            return {"energy": 0, "pace": 0, "orientation": 0, "structure": 0, "expression": 0}
        
        weighted_axes = {"energy": 0.0, "pace": 0.0, "orientation": 0.0, "structure": 0.0, "expression": 0.0}
        total_weight = 0.0
        
        for idx, comp_id in enumerate(components):
            if comp_id not in self.answers:
                continue
            
            # Weight decreases with position: 1.0, 0.5, 0.33, 0.25...
            weight = 1.0 / (idx + 1)
            comp_axes = self.answers[comp_id]["axes"]
            
            for axis in weighted_axes.keys():
                weighted_axes[axis] += comp_axes[axis] * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for axis in weighted_axes.keys():
                weighted_axes[axis] /= total_weight
        
        return weighted_axes
    
    def _rebuild_vector_index(self) -> None:
        """Rebuild ChromaDB collection for vector fallback (Tier 3)."""
        collection_name = "alignments"
        
        print(f"[DEBUG] Deleting old ChromaDB collection...")
        # Delete old collection
        try:
            self._client.delete_collection(name=collection_name)
        except:
            pass
        
        print(f"[DEBUG] Creating new ChromaDB collection...")
        # Create new collection
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        if not self.alignments:
            print(f"[DEBUG] No alignments to index")
            return
        
        print(f"[DEBUG] Preparing {len(self.alignments)} alignment documents for embedding...")
        # Prepare documents for embedding
        documents = []
        ids = []
        metadatas = []
        
        for alignment_id, alignment in self.alignments.items():
            # Create text representation for semantic search
            # Include name, description, and component answer texts
            text_parts = [alignment["name"], alignment["description"]]
            
            for comp_id in alignment["components"]:
                if comp_id in self.answers:
                    text_parts.append(self.answers[comp_id]["text"])
            
            full_text = " ".join(text_parts)
            
            documents.append(full_text)
            ids.append(alignment_id)
            metadatas.append({
                "type": alignment["type"],
                "categories": ",".join(alignment["categories"])
            })
        
        # Add to ChromaDB (it will auto-generate embeddings)
        print(f"[DEBUG] Adding {len(documents)} documents to ChromaDB (generating embeddings - may take 30-60s)...")
        if documents:
            self._collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
        print(f"[DEBUG] ChromaDB indexing complete!")
    
    def query_vector(self, query_text: str, n_results: int = 5, category_filter: Optional[str] = None) -> List[str]:
        """
        Query ChromaDB for similar alignments (Tier 3 fallback).
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            category_filter: Optional category to filter by
            
        Returns:
            List of alignment IDs sorted by similarity
        """
        if not self._collection:
            return []
        
        where = None
        if category_filter:
            where = {"categories": {"$contains": category_filter}}
        
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results["ids"][0] if results["ids"] else []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        return {
            "answers_count": len(self.answers),
            "alignments_count": len(self.alignments),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "csv_path": str(self.csv_path),
            "alignments_by_type": self._count_by_type()
        }
    
    def _count_by_type(self) -> Dict[str, int]:
        """Count alignments by type."""
        counts = {}
        for alignment in self.alignments.values():
            atype = alignment["type"]
            counts[atype] = counts.get(atype, 0) + 1
        return counts
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float, return 0.0 if invalid."""
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _load_dataframe(self, file_path: Path) -> pd.DataFrame:
        """
        Load DataFrame from CSV or Excel file.
        
        Args:
            file_path: Path to CSV or Excel file
            
        Returns:
            Loaded DataFrame
        """
        suffix = file_path.suffix.lower()
        
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
        elif suffix == ".csv":
            return pd.read_csv(file_path)
        else:
            # Try CSV as default
            return pd.read_csv(file_path)
