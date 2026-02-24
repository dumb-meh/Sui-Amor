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
        
        # Make both LLM calls separately
        affirmation_prompt = self._create_affirmation_prompt(request)
        summary_prompt = self._create_quiz_summary_prompt(request)
        
        # Call both (could be parallelized in future if needed)
        affirmation_data = self._get_openai_response(affirmation_prompt, model=self.reasoning_model)
        summary_data = self._get_openai_response(summary_prompt, model="gpt-4o-mini")
        
        if not affirmation_data:
            raise ValueError("Failed to generate affirmations")
        
        if not summary_data:
            raise ValueError("Failed to generate quiz summary")
        
        affirmations = affirmation_data.get("affirmation", [])
        theme = affirmation_data.get("affirmation_theme", "")
        summary = summary_data.get("short_summary_of_quiz", "")
        
        if len(affirmations) != 12:
            raise ValueError(f"Expected 12 affirmations, got {len(affirmations)}")
        
        return affirmation_response(
            affirmation=affirmations, 
            affirmation_theme=theme,
            short_summary_of_quiz=summary
        )

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

    def _create_quiz_summary_prompt(self, request: affirmation_request) -> str:
        """Build the prompt for generating quiz summary (separate LLM call)."""
        
        system_prompt = """You are an expert at analyzing quiz responses and creating thoughtful, personalized summaries.

Your task is to generate a short summary of the user's quiz responses that reflects their values, patterns, and current focus.

You MUST return valid JSON matching this exact structure:
{
  "short_summary_of_quiz": "the summary text here"
}

CRITICAL FORMATTING RULES:
1. Use \\n to indicate where new lines should appear in the summary
2. Use - (dash) instead of • (bullet points) for list items
3. Follow this EXACT structure in your summary:

Paragraph 1: Overall balance and values (2-3 sentences)
\\n\\n
Paragraph 2: How they recharge and what motivates them (2-3 sentences)
\\n\\n
"Your pattern reflects someone who:"
\\n\\n
- First characteristic
\\n
- Second characteristic
\\n
- Third characteristic
\\n
- Fourth characteristic
\\n\\n
Paragraph 3: Current focus and growth direction (1-2 sentences)

EXAMPLE OUTPUT (follow this structure exactly):
{
  "short_summary_of_quiz": "Your results show a balance between steady grounding and growth-oriented energy. You value stability, learning, and personal progress, and you tend to move forward through consistency rather than intensity.\\n\\nYou recharge in quiet or natural environments, where you can think clearly and reconnect with your direction. Creating, helping others, and improving yourself appear to be important sources of motivation.\\n\\nYour pattern reflects someone who:\\n\\n- Seeks meaning and steady progress\\n- Values learning and personal development\\n- Regains energy through calm, structured environments\\n- Moves forward with purpose rather than impulse\\n\\nYour current focus on self-worth and connection suggests a period of strengthening your foundation — building confidence, clarity, and emotional stability as you grow."
}

IMPORTANT:
- Analyze ONLY the quiz data provided
- Be specific and personalized based on their actual answers
- Keep the tone warm, insightful, and affirming
- The summary should be 4-5 short paragraphs total
- Always include the "Your pattern reflects someone who:" section with 4 dash-prefixed items"""

        # Convert quiz data to readable format
        quiz_summary = []
        for item in request.quizdata:
            question_data = {"question": item.question}
            if item.answers:
                question_data["answers"] = item.answers
            if item.sub_questions:
                sub_q_data = []
                for sq in item.sub_questions:
                    sub_q_data.append({
                        "sub_question": sq.sub_question,
                        "sub_answers": sq.sub_answers
                    })
                question_data["sub_questions"] = sub_q_data
            quiz_summary.append(question_data)

        user_payload = {
            "quiz_data": quiz_summary,
            "instructions": "Generate a personalized quiz summary following the exact format and structure specified. Use \\n for line breaks and - for bullet points."
        }

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def _get_openai_response(self, prompt: str, model: str = None) -> Dict[str, Any] | None:
        """Call OpenAI API to generate affirmations or quiz summary."""
        try:
            payload_data = json.loads(prompt)
            system_content = payload_data.get("system", "")
            user_content = json.dumps(payload_data.get("payload", {}), ensure_ascii=False)

            # Use provided model or default to reasoning model
            selected_model = model or self.reasoning_model

            response = self.client.chat.completions.create(
                model=selected_model,
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
