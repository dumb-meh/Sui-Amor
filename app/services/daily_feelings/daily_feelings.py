import hashlib
import openai
from app.core.config import settings
from app.utils.cache_manager import cache_manager


class ChatbotService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        self.prompt_version = "manifex_v4"

    def get_response(self, user_id: str, feeling: str, reason: str, surface: str) -> str:
        cleaned_feeling = (feeling or "").strip()
        cleaned_reason = (reason or "").strip()
        if not cleaned_feeling and not cleaned_reason:
            return "Please share how you feel and why, so I can respond with care."

        normalized_surface = (surface or "").strip().lower()
        if normalized_surface not in {"app", "web"}:
            normalized_surface = "web"

        goal_data = cache_manager.get_user_goal(user_id) or {}
        goal = (goal_data.get("goal") or "").strip()

        scoped_user_id = self._scoped_user_id(user_id, normalized_surface)
        history = cache_manager.get_daily_feelings_history(scoped_user_id) if scoped_user_id else None
        history = history or []
        history_to_send = history[-10:]
        use_response_cache = len(history) == 0

        cache_key = None
        if use_response_cache:
            cache_key = self._response_cache_key(user_id, cleaned_feeling, cleaned_reason, goal, normalized_surface)
            cached = cache_manager.get_cached_response(cache_key)
            if cached:
                return cached

        messages = [{"role": "system", "content": self.create_prompt(normalized_surface, goal, history_to_send)}]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Feeling: {cleaned_feeling or 'not provided'}\n"
                    f"Reason: {cleaned_reason or 'not provided'}"
                ),
            }
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )
        reply = completion.choices[0].message.content.strip()

        if scoped_user_id:
            cache_manager.append_daily_feeling_history(
                scoped_user_id,
                cleaned_feeling,
                cleaned_reason,
                reply,
            )

        if use_response_cache and cache_key:
            cache_manager.set_cached_response(cache_key, reply)

        return reply

    def _scoped_user_id(self, user_id: str, surface: str) -> str:
        if not user_id:
            return ""

        return f"{surface}:{user_id}"

    def _response_cache_key(self, user_id: str, feeling: str, reason: str, goal: str, surface: str) -> str:
        signature = f"{self.prompt_version}|{surface}|{user_id}|{goal}|{feeling}|{reason}"
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    def create_prompt(self, surface: str, goal: str, past_daily_feelings: list) -> str:
        past_lines = []
        for item in past_daily_feelings[-10:]:
            feeling = item.get("feeling", "")
            reason = item.get("reason", "")
            ai_response = item.get("ai_response", "")
            past_lines.append(f"- Feeling: {feeling} | Reason: {reason} | AI: {ai_response}")

        past_section = "\n".join(past_lines) if past_lines else "No previous daily feelings yet."
        goal_section = goal if goal else "No stored goal available."

        return f"""You are Sui Amor, a warm, elegant, and emotionally intelligent assistant for a daily feelings check-in.

User goal: {goal_section}

Recent daily feelings:
{past_section}

Write a short, beautiful response that acknowledges the user's feeling and reason, and gently connects it to their goal when relevant. Reinforce the user's belief in themselves. If the user sounds sad, overwhelmed, or depressed, be especially supportive, grounding, and kind. Keep the tone polished, personal, and concise. Do not mention internal rules, Redis, history storage, or system instructions. Avoid long explanations and avoid generic motivational language.
"""















