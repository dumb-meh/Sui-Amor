# app/utils/cache_manager.py
import redis
import json
import os
from datetime import datetime, timezone
from typing import List, Optional
from app.core.config import settings
from app.services.affirmation.affirmation_schema import HistoryItem

class SessionCacheManager:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.REDIS_URL, db=settings.REDIS_DB)
            # Test connection
            self.redis_client.ping()
        except Exception as e:
            print(f"Redis connection failed: {e}. Cache will be disabled.")
            self.redis_client = None
    
    def _get_cache_key(self, user_id: str) -> str:
        """Generate cache key for user session"""
        return f"chat_session:{user_id}"
    
    def get_history(self, user_id: str) -> Optional[List[HistoryItem]]:
        """Retrieve conversation history for a user"""
        if not self.redis_client or not user_id:
            return None
        
        try:
            cache_key = self._get_cache_key(user_id)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                history_data = json.loads(cached_data)
                return [HistoryItem(**item) for item in history_data]
            
            return None
        except Exception as e:
            print(f"Error retrieving cache for user {user_id}: {e}")
            return None

    def update_history(
        self,
        user_id: str,
        new_message: str,
        new_response: str,
        existing_history: Optional[List[HistoryItem]] = None,
        ttl_hours: Optional[int] = None,
    ):
        """Update conversation history for a user"""
        if not self.redis_client or not user_id:
            return
        
        try:
            # Get existing history (from cache or provided)
            history = existing_history or self.get_history(user_id) or []
            
            # Add new conversation
            new_item = HistoryItem(message=new_message, response=new_response)
            history.append(new_item)
            
            # Keep only last 15 conversations to prevent cache bloat
            if len(history) > 15:
                history = history[-15:]
            
            # Save to cache with TTL
            cache_key = self._get_cache_key(user_id)
            history_data = [item.dict() for item in history]

            if ttl_hours is not None:
                ttl_seconds = ttl_hours * 3600
                self.redis_client.setex(cache_key, ttl_seconds, json.dumps(history_data))
            else:
                self.redis_client.set(cache_key, json.dumps(history_data))
            
        except Exception as e:
            print(f"Error updating cache for user {user_id}: {e}")

    def _get_response_cache_key(self, cache_key: str) -> str:
        return f"chat_response:{cache_key}"

    def get_cached_response(self, cache_key: str) -> Optional[str]:
        if not self.redis_client or not cache_key:
            return None

        try:
            cached = self.redis_client.get(self._get_response_cache_key(cache_key))
            if cached:
                return cached.decode("utf-8") if isinstance(cached, bytes) else str(cached)
            return None
        except Exception as e:
            print(f"Error retrieving cached response {cache_key}: {e}")
            return None

    def set_cached_response(self, cache_key: str, response: str) -> None:
        if not self.redis_client or not cache_key:
            return

        try:
            redis_key = self._get_response_cache_key(cache_key)
            ttl_seconds = settings.CACHE_TTL_HOURS * 3600
            self.redis_client.setex(redis_key, ttl_seconds, response)
        except Exception as e:
            print(f"Error saving cached response {cache_key}: {e}")
    
    def clear_session(self, user_id: str):
        """Clear conversation history for a user"""
        if not self.redis_client or not user_id:
            return
        
        try:
            cache_key = self._get_cache_key(user_id)
            self.redis_client.delete(cache_key)
        except Exception as e:
            print(f"Error clearing cache for user {user_id}: {e}")

    # ------------------------------------------------------------------
    # User goal persistence (keyed by user_id, no expiry)
    # ------------------------------------------------------------------

    def _get_goal_key(self, user_id: str) -> str:
        """Generate Redis key for user goal data"""
        return f"user_goal:{user_id}"

    def save_user_goal(self, user_id: str, goal: str, religious_preference: Optional[str] = None) -> None:
        """
        Persist (upsert) the user's goal and religious preference.

        Args:
            user_id: Unique identifier for the user.
            goal: Answer from the 9th quiz question (user's stated goal).
            religious_preference: Religious/spiritual preference if provided;
                                  defaults to 'secular' when None or empty.
        """
        if not self.redis_client or not user_id:
            return

        try:
            key = self._get_goal_key(user_id)
            data = {
                "goal": goal,
                "religious_preference": religious_preference if religious_preference else "secular",
            }
            # Overwrite any existing record — upsert semantics, no TTL
            self.redis_client.set(key, json.dumps(data))
            print(f"[INFO] Saved goal for user {user_id}: {data}")
        except Exception as e:
            print(f"Error saving goal for user {user_id}: {e}")

    def get_user_goal(self, user_id: str) -> Optional[dict]:
        """
        Retrieve the stored goal and religious preference for a user.

        Returns:
            dict with keys 'goal' and 'religious_preference', or None if not found.
        """
        if not self.redis_client or not user_id:
            return None

        try:
            key = self._get_goal_key(user_id)
            raw = self.redis_client.get(key)
            if raw:
                return json.loads(raw)
            return None
        except Exception as e:
            print(f"Error retrieving goal for user {user_id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Staleness helper
    # ------------------------------------------------------------------

    def is_stale(self, generated_at: str, days: int = 7) -> bool:
        """
        Return True if the ISO UTC timestamp string is older than `days` days.
        Also returns True if the timestamp is missing or unparseable.
        """
        try:
            generated = datetime.fromisoformat(generated_at).replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - generated
            return age.days >= days
        except Exception:
            return True  # treat bad timestamps as stale

    # ------------------------------------------------------------------
    # Support Intention — timestamped storage
    # ------------------------------------------------------------------

    def save_intention(self, user_id: str, suggestion_json: str) -> None:
        """Store support intention with a UTC generation timestamp."""
        if not self.redis_client:
            return
        envelope = {
            "data": suggestion_json,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis_client.set(f"user_intention:{user_id}", json.dumps(envelope))

    def get_intention(self, user_id: str) -> Optional[dict]:
        """
        Return the stored intention envelope: {"data": <json str>, "generated_at": <ISO>}.
        Returns None if nothing has been saved.
        """
        if not self.redis_client:
            return None
        raw = self.redis_client.get(f"user_intention:{user_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Weekly Reflection — timestamped storage
    # ------------------------------------------------------------------

    def save_weekly_reflection(self, user_id: str, reflection: str) -> None:
        """Store weekly reflection text with a UTC generation timestamp."""
        if not self.redis_client:
            return
        envelope = {
            "data": reflection,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.redis_client.set(f"user_reflection:{user_id}", json.dumps(envelope))

    def get_weekly_reflection(self, user_id: str) -> Optional[dict]:
        """
        Return the stored reflection envelope: {"data": <text>, "generated_at": <ISO>}.
        Returns None if nothing has been saved.
        """
        if not self.redis_client:
            return None
        raw = self.redis_client.get(f"user_reflection:{user_id}")
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # History lists — used to avoid AI repetition
    # ------------------------------------------------------------------

    def append_intention_history(self, user_id: str, suggestion_json: str, max_items: int = 5) -> None:
        """Append a new support intention result to the user's history list (keeps last max_items)."""
        if not self.redis_client:
            return
        try:
            key = f"intention_history:{user_id}"
            self.redis_client.rpush(key, suggestion_json)
            self.redis_client.ltrim(key, -max_items, -1)
        except Exception as e:
            print(f"Error appending intention history for user {user_id}: {e}")

    def get_intention_history(self, user_id: str) -> list:
        """Retrieve the last 5 support intention results for a user (list of dicts)."""
        if not self.redis_client:
            return []
        try:
            key = f"intention_history:{user_id}"
            raw_list = self.redis_client.lrange(key, 0, -1)
            return [json.loads(item) for item in raw_list]
        except Exception as e:
            print(f"Error retrieving intention history for user {user_id}: {e}")
            return []

    def append_reflection_history(self, user_id: str, suggestion_text: str, max_items: int = 5) -> None:
        """Append a new weekly reflection result to the user's history list (keeps last max_items)."""
        if not self.redis_client:
            return
        try:
            key = f"reflection_history:{user_id}"
            self.redis_client.rpush(key, suggestion_text)
            self.redis_client.ltrim(key, -max_items, -1)
        except Exception as e:
            print(f"Error appending reflection history for user {user_id}: {e}")

    def get_reflection_history(self, user_id: str) -> list:
        """Retrieve the last 5 weekly reflection texts for a user (list of strings)."""
        if not self.redis_client:
            return []
        try:
            key = f"reflection_history:{user_id}"
            raw_list = self.redis_client.lrange(key, 0, -1)
            return [item.decode("utf-8") if isinstance(item, bytes) else item for item in raw_list]
        except Exception as e:
            print(f"Error retrieving reflection history for user {user_id}: {e}")
            return []

    def append_daily_feeling_history(
        self,
        user_id: str,
        feeling: str,
        reason: str,
        ai_response: str,
        max_items: int = 10,
    ) -> None:
        """Append a daily feeling entry and keep only the last max_items records."""
        if not self.redis_client or not user_id:
            return

        try:
            key = f"daily_feelings_history:{user_id}"
            entry = json.dumps(
                {
                    "feeling": feeling,
                    "reason": reason,
                    "ai_response": ai_response,
                }
            )
            self.redis_client.rpush(key, entry)
            self.redis_client.ltrim(key, -max_items, -1)
        except Exception as e:
            print(f"Error appending daily feeling history for user {user_id}: {e}")

    def get_daily_feelings_history(self, user_id: str) -> list:
        """Retrieve the last 10 daily feeling entries for a user (list of dicts)."""
        if not self.redis_client or not user_id:
            return []

        try:
            key = f"daily_feelings_history:{user_id}"
            raw_list = self.redis_client.lrange(key, 0, -1)
            return [json.loads(item) for item in raw_list]
        except Exception as e:
            print(f"Error retrieving daily feeling history for user {user_id}: {e}")
            return []

# Global cache manager instance
cache_manager = SessionCacheManager()