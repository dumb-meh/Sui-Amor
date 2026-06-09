import json
import os
import openai
from dotenv import load_dotenv

from .support_intention_schema import SupportIntentionRequest, SupportIntentionResponse, IntentionItem
from app.utils.cache_manager import cache_manager

load_dotenv()

STALE_DAYS = 7  # Regenerate after this many days


class SupportIntention:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tag_model = "gpt-4o-mini"

    def generate_support_intention(self, user_id: str) -> SupportIntentionResponse:
        """
        Generate exactly 3 personalized intention items and persist them.

        - Goal is fetched from Redis (saved by the affirmation endpoint).
        - Past 5 support intentions are fetched from Redis to avoid repetition.
        - Result is saved as the current intention (with timestamp) and appended to history.
        """
        # 1. Fetch goal from Redis
        goal_data = cache_manager.get_user_goal(user_id) or {}
        goal = goal_data.get("goal", "")

        # 2. Fetch past 5 intention results from Redis (for variety)
        past_intentions = cache_manager.get_intention_history(user_id)

        # 3. Generate via OpenAI
        response = self._generate(goal, past_intentions)
        if not response:
            raise ValueError("Failed to generate support intention")

        # 4. Persist: save as current (with timestamp) + append to history
        suggestion_json = json.dumps([item.model_dump() for item in response.suggestion])
        cache_manager.save_intention(user_id, suggestion_json)
        cache_manager.append_intention_history(user_id, suggestion_json)

        return response

    def get_current_intention(self, user_id: str) -> tuple[SupportIntentionResponse | None, bool]:
        """
        Return (SupportIntentionResponse | None, is_stale).

        - None means nothing has ever been generated for this user.
        - is_stale=True means the cached content is >= 7 days old and should be regenerated.
        """
        envelope = cache_manager.get_intention(user_id)
        if not envelope:
            return None, True

        stale = cache_manager.is_stale(envelope.get("generated_at", ""), days=STALE_DAYS)

        try:
            items_data = json.loads(envelope["data"])
            items = [IntentionItem(**item) for item in items_data]
            return SupportIntentionResponse(suggestion=items), stale
        except Exception as e:
            print(f"[WARN] Failed to deserialize intention for user {user_id}: {e}")
            return None, True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate(self, goal: str, past_intentions: list) -> SupportIntentionResponse | None:
        prompt = self._create_prompt(goal, past_intentions)
        return self._get_openai_response(prompt)

    def _create_prompt(self, goal: str, past_intentions: list) -> str:
        system_prompt = (
            "You are a compassionate, thoughtful self-reflection guide and coach for Sui Amor.\n\n"
            "Your task is to generate exactly 3 intention items for the user based on their focus goal "
            "and previous support intentions (if any).\n\n"
            "The 3 items MUST have these exact titles (in this order):\n"
            '  1. "Morning Intentions"\n'
            '  2. "Mindful Moments"\n'
            '  3. "Evening Reflection"\n\n'
            "For each item, write a short, warm, and actionable description (1–2 sentences) "
            "that is personalized to the user's goal. The descriptions should:\n"
            "- Feel inspiring, soulful, and encouraging.\n"
            "- Be concise and easy to read at a glance.\n"
            "- Vary meaningfully from any previous support intentions provided.\n\n"
            "You MUST return valid JSON matching this exact structure:\n"
            "{\n"
            '  "suggestion": [\n'
            '    {"title": "Morning Intentions", "description": "..."},\n'
            '    {"title": "Mindful Moments", "description": "..."},\n'
            '    {"title": "Evening Reflection", "description": "..."}\n'
            "  ]\n"
            "}\n\n"
            "Example of a well-formed response (for a goal of 'build confidence'):\n"
            "{\n"
            '  "suggestion": [\n'
            '    {"title": "Morning Intentions", "description": "Read your affirmations aloud and remind yourself of one strength you bring to the world today."},\n'
            '    {"title": "Mindful Moments", "description": "When self-doubt arises, pause, take a breath, and gently return to your intention of growing into your confidence."},\n'
            '    {"title": "Evening Reflection", "description": "Celebrate one moment today where you showed up bravely, and release any judgment with gratitude."}\n'
            "  ]\n"
            "}"
        )

        user_payload = {
            "goal": goal or "general well-being",
            "previous_support_intentions": past_intentions,
            "instructions": (
                "Generate 3 personalized intention descriptions for the fixed titles. "
                "Keep each description short (1–2 sentences), warm, and tailored to the goal. "
                "Ensure the new intentions differ meaningfully from the previous ones provided."
            ),
        }

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def _get_openai_response(self, prompt: str) -> SupportIntentionResponse | None:
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
                raw_items = data.get("suggestion", [])
                if isinstance(raw_items, list) and len(raw_items) == 3:
                    items = [
                        IntentionItem(title=item["title"], description=item["description"])
                        for item in raw_items
                    ]
                    return SupportIntentionResponse(suggestion=items)
        except Exception as e:
            print(f"[ERROR] SupportIntention OpenAI call failed: {e}")
            return None
        return None
