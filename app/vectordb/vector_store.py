import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb


class AlignmentsVectorStore:
    """Wrapper around a persistent Chroma collection for alignment records."""

    def __init__(self, persist_dir: str | Path | None = None, collection_name: str = "alignments") -> None:
        self.persist_dir = Path(persist_dir or Path.cwd() / ".chroma")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    @property
    def collection(self):
        return self._collection

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, *, records: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]]) -> None:
        if not records:
            return
        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []
        for record in records:
            record_id = record.get("id") or str(uuid.uuid4())
            ids.append(record_id)
            metadatas.append({k: v for k, v in record.items() if k not in {"id", "embedding_text"}})
            documents.append(record.get("embedding_text", ""))
        self._collection.upsert(ids=ids, embeddings=list(embeddings), documents=documents, metadatas=metadatas)

    def query(self, *, embedding: Sequence[float], limit: int) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        results = self._collection.query(query_embeddings=[list(embedding)], n_results=limit)
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
        results = self._collection.get()
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
