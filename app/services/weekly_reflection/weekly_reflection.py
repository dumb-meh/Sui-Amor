import json
import os
import openai
from dotenv import load_dotenv

from .weekly_reflection_schema import ReflectionRequest, ReflectionResponse

load_dotenv()


class Reflection:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tag_model = "gpt-4o-mini"

    def generate_weekly_reflection(self, request: ReflectionRequest) -> ReflectionResponse:
        """
        Generate a thoughtful weekly reflection and suggestion based on previous reflections and goal.
        """
        prompt = self.create_prompt(request)
        response = self.get_openai_response(prompt)
        
        if not response:
            return ReflectionResponse(
                suggestion="Take a moment to breathe and appreciate your journey. Reflect on your goals and take one gentle step forward today."
            )
        return response

    def create_prompt(self, request: ReflectionRequest) -> str:
        system_prompt = (
            "You are a compassionate, thoughtful self-reflection guide and coach for Sui Amor.\n\n"
            "Your task is to analyze the user's focus goal and their previous weekly reflections (if any), "
            "and craft a deeply inspiring, highly personalized, and supportive weekly reflection and growth suggestion.\n\n"
            "The reflection should:\n"
            "- Acknowledge their focus/goal.\n"
            "- Connect insights from their previous reflections (if provided) to show their growth trajectory.\n"
            "- Offer gentle, actionable, and soulful advice for the week ahead.\n"
            "- Be written in a warm, poetic, and encouraging tone.\n"
            "- Flow naturally, using paragraphs rather than bullet points, to create a premium, cohesive reading experience.\n\n"
            "You MUST return valid JSON matching this exact structure:\n"
            "{\n"
            '  "suggestion": "Your beautiful, personalized weekly reflection and suggestion text here."\n'
            "}"
        )
        
        user_payload = {
            "goal": request.goal,
            "previous_reflections": request.previous_reflections or [],
            "instructions": "Generate a beautiful weekly reflection and suggestion. Ensure it feels premium, tailored to their goal, and flows beautifully."
        }
        
        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def get_openai_response(self, prompt: str) -> ReflectionResponse | None:
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
        except Exception:
            return None
        return None
