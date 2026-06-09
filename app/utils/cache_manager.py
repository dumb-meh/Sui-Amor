# app/utils/cache_manager.py
import redis
import json
import os
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
    
    def update_history(self, user_id: str, new_message: str, new_response: str, existing_history: Optional[List[HistoryItem]] = None):
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
            ttl_seconds = settings.CACHE_TTL_HOURS * 3600  # Convert hours to seconds
            
            self.redis_client.setex(cache_key, ttl_seconds, json.dumps(history_data))
            
        except Exception as e:
            print(f"Error updating cache for user {user_id}: {e}")
    
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

# Global cache manager instance
cache_manager = SessionCacheManager()