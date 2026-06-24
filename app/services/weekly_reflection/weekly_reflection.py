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
                suggestion="What is one small step you can take this week toward your goal?"
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
            "You are a thoughtful self-reflection guide for Sui Amor, a personal growth platform.\n\n"
            "Your task is to write ONE short, meaningful weekly reflection question for the user.\n\n"
            "RULES:\n"
            "- Write exactly ONE question — nothing else.\n"
            "- The question must be short (one sentence, ideally under 15 words).\n"
            "- It should invite genuine self-inquiry, not give advice or affirmations.\n"
            "- It must feel relevant to the user's current goal (if provided).\n"
            "- It must differ meaningfully from any previous reflection questions provided.\n"
            "- Write it in second person ('you' / 'your'), present or past tense for the week.\n"
            "- Do NOT start with 'I' or write it as a statement — it must be a question.\n\n"
            "Example questions (use these as style reference, do NOT repeat them):\n"
            "- What are you most grateful for this week?\n"
            "- What challenged you this week?\n"
            "- Where did you show up for yourself this week?\n"
            "- What gave you energy this week?\n"
            "- What would you like to improve next week?\n"
            "- What are you proud of this week?\n"
            "- Where did your actions align with your intentions?\n"
            "- What lesson are you taking into next week?\n\n"
            "You MUST return valid JSON matching this exact structure:\n"
            "{\n"
            '  "suggestion": "Your short reflection question here?"\n'
            "}"
        )

        user_payload = {
            "goal": goal or "general personal growth",
            "previous_reflection_questions": past_reflections,
            "instructions": (
                "Generate one short weekly reflection question tailored to the user's goal. "
                "It must be a genuine question (ending with '?'), under 15 words, "
                "and must not repeat any of the previous questions provided."
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
