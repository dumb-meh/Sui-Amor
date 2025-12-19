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
        self._last_quiz_data: List[Dict[str, Any]] = []

    def quiz_evaluation(self, request: QuizEvaluationRequest) -> QuizEvaluationResponse:
        self._last_quiz_data = request.quiz_data or []
        prompt = self.create_prompt()
        raw_results = self.get_openai_response(prompt)
        payload = self._format_vector_results(raw_results)
        return QuizEvaluationResponse(**payload)

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
            qualities = item.get("qualities", [])
            themes = item.get("themes", [])
            
            # Add qualities as profile tags
            if isinstance(qualities, list):
                for quality in qualities:
                    if isinstance(quality, str) and quality.strip():
                        profile_tags_set.add(quality.strip().lower())
            
            # Add themes as profile tags
            if isinstance(themes, list):
                for theme in themes:
                    if isinstance(theme, str) and theme.strip():
                        profile_tags_set.add(theme.strip().lower())
            
            # Extract tags from attributes if they exist
            attributes = item.get("attributes", {})
            if isinstance(attributes, dict):
                for key, value in attributes.items():
                    if isinstance(value, str) and value.strip():
                        # Add attribute values as potential profile tags
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
    


