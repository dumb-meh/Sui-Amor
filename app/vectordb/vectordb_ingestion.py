import io
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import json
import openai
import pandas as pd

from app.core.config import settings
from app.vectordb.vector_store import AlignmentsVectorStore

ALIGNMENT_KEYWORDS = (
    "Synergy",
    "Harmony",
    "Resonance",
    "Polarity",
)


class AlignmentIngestionService:
    """Reads tabular alignment data, extracts records, and stores them in the vector DB."""

    def __init__(self, *, vector_store: AlignmentsVectorStore | None = None, embedding_model: str = "text-embedding-3-large") -> None:
        self.vector_store = vector_store or AlignmentsVectorStore()
        self.embedding_model = embedding_model
        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

    def ingest_file(self, uploaded_file: io.BufferedReader | io.BytesIO | Path, *, reset_collection: bool = True) -> Dict[str, Any]:
        dataframe = self._load_dataframe(uploaded_file)
        all_records: List[Dict[str, Any]] = []
        for row in dataframe.itertuples(index=False):
            all_records.extend(self._parse_row(list(row)))
        if not all_records:
            return {"records": 0}
        if reset_collection:
            self.vector_store.reset()
        self._ensure_unique_ids(all_records)
        embeddings = [self._embed_text(record["embedding_text"]) for record in all_records]
        self.vector_store.upsert(records=all_records, embeddings=embeddings)
        by_type = _count_by_type(all_records)
        return {"records": len(all_records), "by_type": by_type}

    def _embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model=self.embedding_model, input=[text])
        return response.data[0].embedding

    def _load_dataframe(self, uploaded_file: io.BufferedReader | io.BytesIO | Path) -> pd.DataFrame:
        if isinstance(uploaded_file, Path):
            suffix = uploaded_file.suffix.lower()
            if suffix in {".xlsx", ".xls"}:
                return pd.read_excel(uploaded_file)
            return pd.read_csv(uploaded_file)
        if hasattr(uploaded_file, "name") and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        return pd.read_csv(uploaded_file)

    def _parse_row(self, values: List[Any]) -> List[Dict[str, Any]]:
        string_values = [self._clean_str(value) for value in values if isinstance(value, str) and value.strip()]
        if not string_values:
            return []
        context_values: List[str] = []
        alignment_blocks: List[str] = []
        for cell in string_values:
            if "Core Essence:" in cell and any(keyword in cell for keyword in ALIGNMENT_KEYWORDS):
                alignment_blocks.append(cell)
            else:
                context_values.append(cell)
        context = self._build_context(context_values)
        records: List[Dict[str, Any]] = []
        for block in alignment_blocks:
            parsed = self._parse_alignment_block(block, context)
            if parsed:
                records.append(parsed)
        return records

    def _build_context(self, cells: Iterable[str]) -> Dict[str, Any]:
        cells = list(cells)
        base_context: Dict[str, Any] = {
            "realm": cells[0] if cells else "",
            "pillar": cells[1] if len(cells) > 1 else "",
            "environment": cells[2] if len(cells) > 2 else "",
            "themes": _split_list(cells[3]) if len(cells) > 3 else [],
            "qualities": _split_list(cells[4]) if len(cells) > 4 else [],
            "i_do": _split_list(cells[5]) if len(cells) > 5 else [],
            "i_feel": _split_list(cells[6]) if len(cells) > 6 else [],
            "i_am": _split_list(cells[7]) if len(cells) > 7 else [],
        }
        attributes: Dict[str, str] = {}
        for cell in cells[8:]:
            if " - " in cell:
                key, value = cell.split(" - ", 1)
                attributes[key.strip().lower().replace(" ", "_")] = value.strip()
        if attributes:
            base_context["attributes"] = attributes
        return base_context

    def _parse_alignment_block(self, block: str, context: Dict[str, Any]) -> Dict[str, Any] | None:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return None
        title_line = lines[0]
        alignment_type = self._infer_type_from_title(title_line)
        if not alignment_type:
            return None
        display_title = self._extract_display_title(title_line)
        core_essence = self._extract_section(lines, "Core Essence:")
        narrative_label = self._narrative_label(alignment_type)
        description = self._extract_section(lines, narrative_label)
        record_id = self._build_identifier(alignment_type, context, display_title)
        embedding_segments = [
            context.get("realm", ""),
            context.get("environment", ""),
            display_title,
            core_essence,
            description,
        ]
        embedding_text = "\n".join([segment for segment in embedding_segments if segment])
        metadata = {
            "id": record_id,
            "type": alignment_type,
            "title": display_title,
            "full_title": title_line,
            "description": description,
            "core_essence": core_essence,
            "realm": context.get("realm"),
            "environment": context.get("environment"),
            "themes": _stringify_metadata(context.get("themes")),
            "qualities": _stringify_metadata(context.get("qualities")),
            "attributes": _stringify_metadata(context.get("attributes")),
            "embedding_text": embedding_text,
        }
        return metadata

    def _ensure_unique_ids(self, records: List[Dict[str, Any]]) -> None:
        seen: Dict[str, int] = {}
        for record in records:
            base_id = record.get("id") or "record"
            count = seen.get(base_id, 0) + 1
            seen[base_id] = count
            if count == 1:
                continue
            record["id"] = f"{base_id}-{count}"

    def query_by_text(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Embed a text query and return nearest records from the vector store."""
        if not text or not text.strip():
            return []
        embedding = self._embed_text(text)
        return self.vector_store.query(embedding=embedding, limit=limit)

    def _infer_type_from_title(self, title: str) -> str | None:
        title = title.lower()
        if "synergy" in title:
            return "synergies"
        if "harmony" in title:
            return "harmonies"
        if "resonance" in title:
            return "resonances"
        if "polarity" in title:
            return "polarities"
        return None

    def _extract_display_title(self, title_line: str) -> str:
        if "–" in title_line:
            return title_line.split("–", 1)[1].strip()
        if "-" in title_line:
            return title_line.split("-", 1)[1].strip()
        return title_line.strip()

    def _extract_section(self, lines: List[str], label: str) -> str:
        collected: List[str] = []
        capture = False
        for line in lines[1:]:
            if line.startswith(label):
                collected.append(line.split(":", 1)[1].strip())
                capture = True
                continue
            if capture and any(line.startswith(keyword) for keyword in ("Core Essence:", "Synergy Text:", "Harmony Text:", "Resonance Text:", "Polarity Text:")):
                break
            if capture:
                collected.append(line)
        return " ".join(collected).strip()

    def _narrative_label(self, alignment_type: str) -> str:
        if alignment_type == "synergies":
            return "Synergy Text:"
        if alignment_type == "harmonies":
            return "Harmony Text:"
        if alignment_type == "resonances":
            return "Resonance Text:"
        return "Polarity Text:"

    def _build_identifier(self, alignment_type: str, context: Dict[str, Any], title: str) -> str:
        slug_components = [alignment_type.rstrip("s"), context.get("realm", ""), context.get("environment", ""), title]
        slug = "-".join(_slugify(part) for part in slug_components if part)
        return slug or _slugify(title)

    def _clean_str(self, value: str) -> str:
        return value.strip().strip('"').replace("\u2014", "—")


def _split_list(text: str | None) -> List[str]:
    if not text:
        return []
    parts = [part.strip() for part in re.split(r",|;", text) if part.strip()]
    return parts


def _slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower())
    return text.strip("-")


def _count_by_type(records: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {key: 0 for key in ("synergies", "harmonies", "resonances", "polarities")}
    for record in records:
        record_type = record.get("type")
        if record_type in counts:
            counts[record_type] += 1
    return counts


def _stringify_metadata(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    return str(value)
