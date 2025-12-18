import json
import os
from pathlib import Path
from typing import List

import openai
from dotenv import load_dotenv

from .affirmation_schema import affirmation_response

load_dotenv()


class _AffirmationCache:
    def __init__(self, directory: Path, max_items: int) -> None:
        self.directory = directory
        self.max_items = max_items
        self.directory.mkdir(parents=True, exist_ok=True)

    def _safe_user_file(self, user_id: str | None) -> Path:
        if not user_id:
            return self.directory / "history.json"
        sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in user_id)
        return self.directory / f"{sanitized}.json"

    def _read(self, user_id: str | None) -> List[str]:
        file_path = self._safe_user_file(user_id)
        if file_path.exists():
            try:
                return json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def is_recent(self, text: str, user_id: str | None) -> bool:
        history = self._read(user_id)
        return text in history[-self.max_items :]

    def remember(self, text: str, user_id: str | None) -> None:
        file_path = self._safe_user_file(user_id)
        history = self._read(user_id)
        history.append(text)
        history = history[-self.max_items :]
        file_path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


class Affirmation:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        base_cache_dir = Path(__file__).resolve().parent / "cache"
        self._daily_cache = _AffirmationCache(base_cache_dir / "daily", max_items=15)
        self._monthly_cache = _AffirmationCache(base_cache_dir / "monthly", max_items=5)

    def get_daily_affirmation(self, user_id: str | None = None) -> affirmation_response:
        prompt = self.create_prompt()
        affirmation = self._generate_unique_affirmation(prompt, self._daily_cache, user_id)
        return affirmation_response(affirmation=affirmation)

    def create_prompt(self) -> str:
        return (
            "Write one daily affirmation for a Sui Amor guest. "
            "Keep it first-person, 10 words or fewer, and centred on mindful self-confidence. "
            "Use warm, modern language and return only the affirmation sentence."
        )

    def get_monthly_affirmation(self, user_id: str | None = None) -> affirmation_response:
        prompt = self.create_monthly_prompt()
        affirmation = self._generate_unique_affirmation(prompt, self._monthly_cache, user_id)
        return affirmation_response(affirmation=affirmation)

    def create_monthly_prompt(self) -> str:
        return (
            "Compose one monthly affirmation for the Sui Amor journal card. "
            "Use first-person voice, invite courageous softness, and keep the message within one elegant sentence wrapped in quotation marks."
        )
    
    def get_openai_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content.strip()

    def _generate_unique_affirmation(
        self,
        prompt: str,
        cache: _AffirmationCache,
        user_id: str | None,
    ) -> str:
        candidate = ""
        for _ in range(5):
            candidate = self.get_openai_response(prompt)
            if candidate and not cache.is_recent(candidate, user_id):
                cache.remember(candidate, user_id)
                return candidate
        if candidate:
            cache.remember(candidate, user_id)
        return candidate
