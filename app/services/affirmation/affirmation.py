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
        self._file = self.directory / "history.json"

    def _read(self) -> List[str]:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return []
        return []

    def is_recent(self, text: str) -> bool:
        history = self._read()
        return text in history[-self.max_items :]

    def remember(self, text: str) -> None:
        history = self._read()
        history.append(text)
        history = history[-self.max_items :]
        self._file.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")


class Affirmation:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        base_cache_dir = Path(__file__).resolve().parent / "cache"
        self._daily_cache = _AffirmationCache(base_cache_dir / "daily", max_items=15)
        self._monthly_cache = _AffirmationCache(base_cache_dir / "monthly", max_items=5)

    def get_daily_affirmation(self) -> affirmation_response:
        prompt = self.create_prompt()
        affirmation = self._generate_unique_affirmation(prompt, self._daily_cache)
        return affirmation_response(affirmation=affirmation)

    def create_prompt(self) -> str:
        return (
            "Write one daily affirmation for a Sui Amor guest. "
            "Keep it first-person, 10 words or fewer, and centred on mindful self-confidence. "
            "Use warm, modern language and return only the affirmation sentence."
        )

    def get_monthly_affirmation(self, input_data: dict) -> affirmation_response:
        prompt = self.create_monthly_prompt()
        affirmation = self._generate_unique_affirmation(prompt, self._monthly_cache)
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

    def _generate_unique_affirmation(self, prompt: str, cache: _AffirmationCache) -> str:
        candidate = ""
        for _ in range(5):
            candidate = self.get_openai_response(prompt)
            if candidate and not cache.is_recent(candidate):
                cache.remember(candidate)
                return candidate
        if candidate:
            cache.remember(candidate)
        return candidate
