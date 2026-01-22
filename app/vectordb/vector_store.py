import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import chromadb


class AlignmentsVectorStore:
    """Wrapper around a persistent Chroma collection for alignment records."""

    def __init__(self, persist_dir: str | Path | None = None, collection_name: str = "alignments") -> None:
        self.persist_dir = Path(persist_dir or Path.cwd() / ".chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None

    @property
    def collection(self):
        return self._get_collection()

    def reset(self) -> None:
        """Drop and recreate the collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
        except Exception:
            # Ignore if the collection does not already exist
            pass
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        return self._collection

    def _run_with_collection(self, operation: Callable):
        try:
            return operation(self._get_collection())
        except Exception as error:
            message = str(error).lower()
            if "does not exist" in message or "not found" in message:
                self._collection = None
                return operation(self._get_collection())
            raise

    def upsert(self, *, records: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]]) -> None:
        if not records:
            return
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []
        for record in records:
            record_id = record.get("id") or str(uuid.uuid4())
            ids.append(record_id)
            metadatas.append(_normalize_metadata(record))
            documents.append(record.get("embedding_text", ""))
        self._run_with_collection(lambda collection: collection.upsert(ids=ids, embeddings=list(embeddings), documents=documents, metadatas=metadatas))

    def query(self, *, embedding: Sequence[float], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        results = self._run_with_collection(lambda collection: collection.query(query_embeddings=[list(embedding)], n_results=limit))
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        payload: List[Dict[str, Any]] = []
        for idx, metadata in zip(ids, metadatas):
            entry: Dict[str, Any] = dict(metadata)
            entry["id"] = idx
            payload.append(entry)
        for idx, distance in zip(range(len(payload)), distances):
            payload[idx]["distance"] = distance
        return payload

    def get_all(self) -> List[Dict[str, Any]]:
        results = self._run_with_collection(lambda collection: collection.get())
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        payload: List[Dict[str, Any]] = []
        for idx, metadata in enumerate(metadatas):
            entry: Dict[str, Any] = dict(metadata)
            entry["id"] = ids[idx]
            entry["embedding_text"] = documents[idx]
            payload.append(entry)
        return payload

    def get_by_type(self, alignment_type: str) -> List[Dict[str, Any]]:
        """Retrieve all records of a specific alignment type."""
        results = self._run_with_collection(
            lambda collection: collection.get(
                where={"type": alignment_type}
            )
        )
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        payload: List[Dict[str, Any]] = []
        for idx, metadata in enumerate(metadatas):
            entry: Dict[str, Any] = dict(metadata)
            entry["id"] = ids[idx]
            entry["embedding_text"] = documents[idx]
            payload.append(entry)
        return payload


def _normalize_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if key in {"id", "embedding_text"}:
            continue
        normalized[key] = _normalize_value(value)
    return normalized


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)
