import json
import os
import openai
from dotenv import load_dotenv

from .support_intention_schema import SupportIntentionRequest, SupportIntentionResponse, IntentionItem

load_dotenv()



class SupportIntention:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tag_model = "gpt-4o-mini"

    def generate_support_intention(self, request: SupportIntentionRequest) -> SupportIntentionResponse:
        """
        Generate exactly 3 personalized intention items based on the user's goal
        and previous support intentions.
        """
        prompt = self.create_prompt(request)
        return self.get_openai_response(prompt)

    def create_prompt(self, request: SupportIntentionRequest) -> str:
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
            "goal": request.goal,
            "previous_support_intentions": [
                item.model_dump() for item in (request.previous_support_intentions or [])
            ],
            "instructions": (
                "Generate 3 personalized intention descriptions for the fixed titles. "
                "Keep each description short (1–2 sentences), warm, and tailored to the goal."
            ),
        }

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def get_openai_response(self, prompt: str) -> SupportIntentionResponse | None:
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
        except Exception:
            return None
        return None
