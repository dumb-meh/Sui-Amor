import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

from app.vectordb.vector_store import AlignmentsVectorStore

from .quiz_evaluation_schema import QuizEvaluationRequest, QuizEvaluationResponse

load_dotenv()


class QuizEvaluation:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = AlignmentsVectorStore()
        self.max_results = 12
        self.reasoning_model = "gpt-4.1-mini"
        self._last_quiz_data: List[Dict[str, Any]] = []

    def quiz_evaluation(self, request: QuizEvaluationRequest) -> QuizEvaluationResponse:
        self._last_quiz_data = request.quiz_data or []
        prompt = self.create_prompt(request)
        vector_results = self.get_openai_response(prompt)
        payload = self._reason_over_results(prompt, vector_results)
        if not payload or not self._has_recommendations(payload):
            payload = self._format_vector_results(vector_results)
        shaped = self._ensure_response_shape(payload)
        return QuizEvaluationResponse(**shaped)

    def create_prompt(self, request: QuizEvaluationRequest | None = None) -> str:
        data = request.quiz_data if request else self._last_quiz_data
        lines: List[str] = []
        for item in data:
            if isinstance(item, dict):
                for question, answer in item.items():
                    question_text = str(question).strip()
                    answer_text = str(answer).strip()
                    if question_text and answer_text:
                        lines.append(f"{question_text}: {answer_text}")
                    elif question_text:
                        lines.append(question_text)
                    elif answer_text:
                        lines.append(answer_text)
            else:
                text = str(item).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines).strip()

    def get_openai_response(self, prompt: str) -> List[Dict[str, Any]]:
        if not prompt:
            return []
        embedding = self._embed_text(prompt)
        return self.vector_store.query(embedding=embedding, limit=self.max_results)

    def _embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model="text-embedding-3-large", input=[text])
        return response.data[0].embedding

    def _reason_over_results(self, quiz_prompt: str, results: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not results:
            return None
        candidates: List[Dict[str, Any]] = []
        for item in results:
            candidates.append(
                {
                    "id": item.get("id", ""),
                    "type": item.get("type", ""),
                    "title": item.get("title") or item.get("full_title") or "",
                    "description": item.get("description", ""),
                    "qualities": self._coerce_list(item.get("qualities")),
                    "themes": self._coerce_list(item.get("themes")),
                    "attributes": self._coerce_dict(item.get("attributes")),
                }
            )
        system_prompt = (
            "You analyze quiz answers and candidate Sui Amor alignments to curate recommendations. "
            "Always produce valid JSON matching the schema with keys synergies, harmonies, resonances, polarities (each with an items list) and profile_tags. "
            "Prefer providing at least one item per group when suitable candidates exist."
        )
        user_payload = {
            "quiz_answers": quiz_prompt,
            "candidates": candidates,
            "instructions": {
                "select_items": "Pick up to 3 relevant items per alignment type ordered by fit. If no candidates match, leave the list empty.",
                "fields_required": ["id", "title", "description"],
                "profile_tags": "Return a lowercase list of distinctive qualities, themes, or attributes that characterize the user based on the chosen recommendations.",
            },
        }
        try:
            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception:
            return None
        return None

    def _format_vector_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        profile_tags_set = set()
        
        for item in results:
            group = item.get("type")
            if group not in {"synergies", "harmonies", "resonances", "polarities"}:
                continue
            grouped[group].append(
                {
                    "id": item.get("id", ""),
                    "title": item.get("title") or item.get("full_title") or "",
                    "description": item.get("description", ""),
                }
            )
            
            # Extract profile tags from qualities, themes, and other relevant fields
            qualities = self._coerce_list(item.get("qualities"))
            themes = self._coerce_list(item.get("themes"))
            
            # Add qualities as profile tags
            for quality in qualities:
                if quality.strip():
                    profile_tags_set.add(quality.strip().lower())
            
            # Add themes as profile tags
            for theme in themes:
                if theme.strip():
                    profile_tags_set.add(theme.strip().lower())
            
            # Extract tags from attributes if they exist
            attributes = self._coerce_dict(item.get("attributes"))
            for value in attributes.values():
                if isinstance(value, str) and value.strip():
                    profile_tags_set.add(value.strip().lower())
        
        # Convert set to sorted list for consistent output
        profile_tags = sorted(list(profile_tags_set))
        
        return {
            "synergies": {"items": grouped.get("synergies", [])},
            "harmonies": {"items": grouped.get("harmonies", [])},
            "resonances": {"items": grouped.get("resonances", [])},
            "polarities": {"items": grouped.get("polarities", [])},
            "profile_tags": profile_tags,
        }

    def _ensure_response_shape(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        template = {
            "synergies": {"items": []},
            "harmonies": {"items": []},
            "resonances": {"items": []},
            "polarities": {"items": []},
            "profile_tags": [],
        }
        result: Dict[str, Any] = {}
        for key, default_value in template.items():
            if key == "profile_tags":
                tags = payload.get(key, [])
                if not isinstance(tags, list):
                    tags = [str(tags)] if tags else []
                result[key] = [str(tag).lower() for tag in tags if str(tag).strip()]
                continue
            section = payload.get(key, {})
            items = []
            if isinstance(section, dict):
                items = section.get("items", [])
            if not isinstance(items, list):
                items = []
            normalized_items: List[Dict[str, str]] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                normalized_items.append(
                    {
                        "id": str(item.get("id", "")),
                        "title": str(item.get("title", "")),
                        "description": str(item.get("description", "")),
                    }
                )
            result[key] = {"items": normalized_items}
        return result

    def _coerce_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
            return [segment.strip() for segment in value.split(",") if segment.strip()]
        return []

    def _coerce_dict(self, value: Any) -> Dict[str, str]:
        if isinstance(value, dict):
            return {str(key): str(val) for key, val in value.items() if str(val).strip()}
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    return {str(key): str(val) for key, val in parsed.items() if str(val).strip()}
            except json.JSONDecodeError:
                pass
        return {}
    


