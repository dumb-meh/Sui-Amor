import json
import os
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
        self._last_quiz_data = [answer.model_dump() for answer in request.answers] if request.answers else []
        prompt = self.create_prompt(request)
        vector_results = self.get_openai_response(prompt)
        payload = self._reason_over_results(prompt, vector_results)
        if not payload:
            raise ValueError("Quiz reasoning returned no response")
        shaped = self._ensure_response_shape(payload)
        self._assert_recommendations(shaped)
        return QuizEvaluationResponse(**shaped)

    def create_prompt(self, request: QuizEvaluationRequest | None = None) -> str:
        data = [answer.model_dump() for answer in request.answers] if request and request.answers else self._last_quiz_data
        return json.dumps({"answers": data}, ensure_ascii=False)

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

        system_prompt = """You are an expert Sui Amor guide. Given quiz answers and candidate alignments, craft tailored combinations.

You will receive quiz data containing an array of answers. Each answer object has a question field. The user's responses can be provided in two ways:
1. Simple answers: An 'answers' array containing one or more string values the user selected
2. Hierarchical answers: A 'sub_questions' array where each sub-question has its own 'sub_question' text and 'sub_answers' array showing what the user selected within that category

Some questions may have only simple answers, some only sub-questions, and both fields are optional.

You MUST return valid JSON matching this exact structure:

{
    "synergies": {"items": [{"id": "...", "title": "...", "description": "..."}]},
    "harmonies": {"items": [{"id": "...", "title": "...", "description": "..."}]},
    "resonances": {"items": [{"id": "...", "title": "...", "description": "..."}]},
    "polarities": {"items": [{"id": "...", "title": "...", "description": "..."}]},
    "profile_tags": ["tag1", "tag2", "tag3"]
}

CRITICAL RULES:
1. TITLE FORMATTING: Always clean up titles. Remove prefixes like "Red –", "Blue –", color names, or any other prefixes. Keep only the core meaningful title (e.g., "Red – Harmony of Inner Peace & Balance" becomes "Harmony of Inner Peace & Balance").

2. ALIGNMENT TYPE ACCURACY: 
     - Synergies = complementary combinations that enhance each other
     - Harmonies = balanced, peaceful alignments
     - Resonances = deep emotional or vibrational connections
     - Polarities = contrasting yet complementary opposites
   
3. APPROPRIATE TITLES: Each item's title MUST match its alignment type. If you're placing an item in "resonances", ensure the title says "Resonance of..." NOT "Synergy of...". If you're placing an item in "polarities", ensure the title says "Polarity of..." NOT "Synergy of...".

4. ONLY RETURN ALIGNMENTS THAT EXIST IN THE PROVIDED CANDIDATES: Do NOT create or invent new alignments. Only include items that are present in the candidates list (vector search results). If there are no suitable candidates for a type, leave that section empty or omit it.

5. OPTIONAL SECTIONS: All sections are optional. Only return items that are truly applicable and meaningful. It's better to omit a section than force irrelevant items into it.

6. Use the exact id from candidates when selecting from the provided list.

Limit profile_tags to 10 concise lowercase tags that capture the user's vibe based on the quiz data and selected alignments."""
        user_payload = {
            "quiz_prompt": quiz_prompt,
            "quiz_data": self._last_quiz_data,
            "candidates": candidates,
            "instructions": {
                "title_cleanup": "ALWAYS remove prefixes like 'Red –', 'Blue –', color names, or any other prefixes from titles. Keep only the meaningful core title.",
                "alignment_accuracy": "Ensure each item's title matches its alignment type. Resonances should have 'Resonance of...' titles, Polarities should have 'Polarity of...' titles, etc.",
                "selection": "Choose up to 3 items per alignment type from candidates. If no suitable candidates exist for a type, leave that section empty or omit it.",
                "optional_sections": "All sections are optional. Only include items that are truly meaningful and applicable. Quality over forced quantity.",
                "profile_tags": "Return up to 10 concise lowercase tags that capture the user's vibe based on quiz answers and selected alignments.",
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
                unique_tags: List[str] = []
                seen = set()
                for tag in tags:
                    normalized = str(tag).strip().lower()
                    if not normalized or normalized in seen:
                        continue
                    unique_tags.append(normalized)
                    seen.add(normalized)
                    if len(unique_tags) >= 10:
                        break
                result[key] = unique_tags
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

    def _assert_recommendations(self, payload: Dict[str, Any]) -> None:
        # Removed assertion - all sections are now optional
        # At least one section should have items, but we won't enforce it
        pass

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
    


