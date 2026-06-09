import json
import os
import openai
from dotenv import load_dotenv

from .weekly_reflection_schema import ReflectionRequest, ReflectionResponse
from app.utils.cache_manager import cache_manager

load_dotenv()

STALE_DAYS = 7  # Regenerate after this many days


class Reflection:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tag_model = "gpt-4o-mini"

    def generate_weekly_reflection(self, user_id: str) -> ReflectionResponse:
        """
        Generate a thoughtful weekly reflection and persist it.

        - Goal is fetched from Redis (saved by the affirmation endpoint).
        - Past 5 reflections are fetched from Redis to avoid repetition.
        - Result is saved as the current reflection (with timestamp) and appended to history.
        """
        # 1. Fetch goal from Redis
        goal_data = cache_manager.get_user_goal(user_id) or {}
        goal = goal_data.get("goal", "")

        # 2. Fetch past 5 reflection texts from Redis (for variety)
        past_reflections = cache_manager.get_reflection_history(user_id)

        # 3. Generate via OpenAI
        response = self._generate(goal, past_reflections)
        if not response:
            response = ReflectionResponse(
                suggestion=(
                    "Take a moment to breathe and appreciate your journey. "
                    "Reflect on your goals and take one gentle step forward today."
                )
            )

        # 4. Persist: save as current (with timestamp) + append to history
        cache_manager.save_weekly_reflection(user_id, response.suggestion)
        cache_manager.append_reflection_history(user_id, response.suggestion)

        return response

    def get_current_reflection(self, user_id: str) -> tuple[ReflectionResponse | None, bool]:
        """
        Return (ReflectionResponse | None, is_stale).

        - None means nothing has ever been generated for this user.
        - is_stale=True means the cached content is >= 7 days old and should be regenerated.
        """
        envelope = cache_manager.get_weekly_reflection(user_id)
        if not envelope:
            return None, True

        stale = cache_manager.is_stale(envelope.get("generated_at", ""), days=STALE_DAYS)
        return ReflectionResponse(suggestion=envelope["data"]), stale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, goal: str, past_reflections: list) -> ReflectionResponse | None:
        prompt = self._create_prompt(goal, past_reflections)
        return self._get_openai_response(prompt)

    def _create_prompt(self, goal: str, past_reflections: list) -> str:
        system_prompt = (
            "You are a compassionate, thoughtful self-reflection guide and coach for Sui Amor.\n\n"
            "Your task is to analyze the user's focus goal and their previous weekly reflections (if any), "
            "and craft a deeply inspiring, highly personalized, and supportive weekly reflection and growth suggestion.\n\n"
            "The reflection should:\n"
            "- Acknowledge their focus/goal.\n"
            "- Connect insights from their previous reflections (if provided) to show their growth trajectory.\n"
            "- Offer gentle, actionable, and soulful advice for the week ahead.\n"
            "- Be written in a warm, poetic, and encouraging tone.\n"
            "- Flow naturally, using paragraphs rather than bullet points, to create a premium, cohesive reading experience.\n"
            "- Differ meaningfully from any previous reflections provided.\n\n"
            "You MUST return valid JSON matching this exact structure:\n"
            "{\n"
            '  "suggestion": "Your beautiful, personalized weekly reflection and suggestion text here."\n'
            "}"
        )

        user_payload = {
            "goal": goal or "general well-being",
            "previous_reflections": past_reflections,
            "instructions": (
                "Generate a beautiful weekly reflection and suggestion. Ensure it feels premium, "
                "tailored to their goal, and flows beautifully. "
                "It must differ meaningfully from the previous reflections provided."
            ),
        }

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def _get_openai_response(self, prompt: str) -> ReflectionResponse | None:
        try:
            payload_data = json.loads(prompt)
            system_content = payload_data.get("system", "")
            user_content = json.dumps(payload_data.get("payload", {}), ensure_ascii=False)

            response = self.client.chat.completions.create(
                model=self.tag_model,
                temperature=0.8,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )

            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                suggestion = data.get("suggestion", "")
                if suggestion:
                    return ReflectionResponse(suggestion=suggestion)
        except Exception as e:
            print(f"[ERROR] WeeklyReflection OpenAI call failed: {e}")
            return None
        return None
