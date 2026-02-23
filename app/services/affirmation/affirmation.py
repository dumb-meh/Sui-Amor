import json
import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv

from .affirmation_schema import affirmation_request, affirmation_response

load_dotenv()


class Affirmation:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reasoning_model = "gpt-4.1"

    def generate_affirmations(self, request: affirmation_request) -> affirmation_response:
        """Generate 12 affirmations based on quiz data and alignments, avoiding past themes."""
        prompt = self._create_affirmation_prompt(request)
        response_data = self._get_openai_response(prompt)
        
        if not response_data:
            raise ValueError("Failed to generate affirmations")
        
        affirmations = response_data.get("affirmation", [])
        theme = response_data.get("affirmation_theme", "")
        
        if len(affirmations) != 12:
            raise ValueError(f"Expected 12 affirmations, got {len(affirmations)}")
        
        return affirmation_response(affirmation=affirmations, affirmation_theme=theme)

    def _create_affirmation_prompt(self, request: affirmation_request) -> str:
        """Build the prompt for generating affirmations, dynamically based on affirmation_type."""

        # Build preference context for the system prompt
        preference_instructions = []
        if request.religious_or_spritual_preference:
            preference_instructions.append(
                f"- RELIGIOUS/SPIRITUAL: The user identifies with {request.religious_or_spritual_preference} spirituality. "
                f"Incorporate {request.religious_or_spritual_preference} values, principles, and spiritual concepts naturally into the affirmations. "
                f"Use language and imagery that aligns with {request.religious_or_spritual_preference} tradition."
            )
        else:
            preference_instructions.append("- RELIGIOUS/SPIRITUAL: Keep affirmations spiritually neutral.")

        if request.holiday_preference:
            preference_instructions.append(
                f"- HOLIDAY: The user celebrates {request.holiday_preference}. "
                f"Incorporate themes of {request.holiday_preference} such as celebration, renewal, gratitude, or community into the affirmations. "
                f"Use imagery and symbolism associated with {request.holiday_preference}."
            )
        else:
            preference_instructions.append("- HOLIDAY: Keep affirmations timeless and universal.")

        if request.astrology_preference:
            preference_instructions.append(
                f"- ASTROLOGY: The user resonates with {request.astrology_preference} energy. "
                f"Weave in {request.astrology_preference} astrological traits, strengths, and characteristics. "
                f"Consider the elemental nature, ruling planets, and symbolic themes of {request.astrology_preference}."
            )
        else:
            preference_instructions.append("- ASTROLOGY: Avoid astrological references.")

        preference_text = "\n".join(preference_instructions)

        # Dynamic affirmation style instructions
        if request.affirmation_type == "freedom":
            affirmation_style = (
                "Each affirmation must consist of exactly 3 lines: three full flowing statements, each sentence on a new line, separated by \\n.\n\n"
                "Example:\n\n"
                "May clarity rise within me when uncertainty begins to speak.\\n"
                "Each step forward strengthens the voice that believes in me.\\n"
                "Confidence grows as I honor my own inner knowing.\\n\n"
                "or\n\n"
                "Courage expands each time I move toward the unknown.\\n"
                "Fear loosens its grip when I choose forward motion.\\n"
                "Strength reveals itself through action, breath by breath.\\n"
            )
        else:  # default to structured
            affirmation_style = (
                "Each affirmation must consist of exactly 5 lines: two full guidance statements (reflective or directional, using 'I will' or 'May I'), each on a new line, followed by three short 'I am' declarations, each on a new line. Each sentence must be on a new line, separated by \\n.\n\n"
                "Example:\n\n"
                "I will trust the quiet wisdom within me, even when uncertainty appears.\\n"
                "I will move forward with clarity, knowing my path becomes stronger with action.\\n"
                "I am capable.\\n"
                "I am certain.\\n"
                "I am confident.\\n\n"
                "Tone variation example:\n\n"
                "May I remember the strength that lives within me, even when my confidence feels quiet.\\n"
                "May I trust my inner voice to guide me forward with clarity and steady courage.\\n"
                "I am capable.\\n"
                "I am certain.\\n"
                "I am enough.\\n"
            )

        system_prompt = f"""You are an expert Sui Amor affirmation creator. Your task is to generate exactly 12 affirmations that form a cohesive set based on the user's quiz data and their selected alignments.

You MUST return valid JSON matching this exact structure:
{{
  "affirmation": ["affirmation 1", "affirmation 2", ..., "affirmation 12"],
  "affirmation_theme": "2-3 word theme"
}}

CRITICAL RULES:
1. CONSISTENCY: All 12 affirmations must be thematically consistent and form a cohesive set around the affirmation_theme.

2. THEME CREATION: The affirmation_theme should be 2-3 words that capture the essence of all 12 affirmations. Examples: "Inner Strength", "Creative Flow", "Peaceful Balance", "Joyful Courage".

3. NO REPETITION: You MUST avoid using any theme or affirmations that appear in the past_theme or past_affirmations provided in the request. Be creative and generate fresh content.

4. AFFIRMATION STYLE: {affirmation_style}

5. ALIGNMENT INTEGRATION: Consider the user's synergies, harmonies, resonances, and polarities when crafting affirmations. Let their selected alignments inspire the theme and tone.

6. PROFILE AWARENESS: Use existing_profile_tags to understand the user's vibe and ensure affirmations resonate with their personality.

7. PREFERENCE HANDLING - FOLLOW THESE EXACTLY:
{preference_text}

IMPORTANT: These preferences are MANDATORY when provided. You MUST incorporate them meaningfully into the affirmations, not just mention them superficially."""

        user_payload = {
            "existing_profile_tags": request.existing_profile_tags or [],
            "synergies": request.synergies or {},
            "harmonies": request.harmonies or {},
            "resonances": request.resonances or {},
            "polarities": request.polarities or {},
            "past_theme": request.past_theme or [],
            "past_affirmations": request.past_affirmations or [],
            "instructions": "Generate exactly 12 affirmations with a 2-3 word theme. Ensure NO repetition of past themes or affirmations. All 12 affirmations must be thematically consistent.",
            "affirmation_type": request.affirmation_type,
        }

        # Add preferences to payload if they exist
        if request.religious_or_spritual_preference:
            user_payload["religious_or_spiritual_preference"] = request.religious_or_spritual_preference
        if request.holiday_preference:
            user_payload["holiday_preference"] = request.holiday_preference
        if request.astrology_preference:
            user_payload["astrology_preference"] = request.astrology_preference

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def _get_openai_response(self, prompt: str) -> Dict[str, Any] | None:
        """Call OpenAI API to generate affirmations."""
        try:
            payload_data = json.loads(prompt)
            system_content = payload_data.get("system", "")
            user_content = json.dumps(payload_data.get("payload", {}), ensure_ascii=False)

            response = self.client.chat.completions.create(
                model=self.reasoning_model,
                temperature=0.9,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
        except Exception:
            return None
        return None
