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
        
        # Q7 (Restoration) energy weights
        self.q7_weights = {
            "Meditating": 25,
            "Reading": 30,
            "Helping Others": 50,
            "Learning": 55,
            "Creating": 60,
            "Planning": 60,
            "Traveling": 75,
            "Exercising": 90
        }
        
        # Q8 (Emotional Pattern) energy weights
        self.q8_weights = {
            "Tired": 20,
            "Calm": 30,
            "Sensitive": 35,
            "Uncertain": 50,
            "Grateful": 50,
            "Hopeful": 60,
            "Excited": 75,
            "Driven": 85
        }
        
        # Secondary scent modifiers by goal
        self.goal_modifiers = {
            "Focus": {"Low": "Frankincense", "High": "Peppermint"},
            "Calm": {"Low": "Vanilla", "High": "Bergamot"},
            "Love": {"Low": "Copaiba", "High": "Neroli"},
            "Joy": {"Low": "Vanilla", "High": "Lime"},
            "Self Worth": {"Low": "Myrrh", "High": "Clary Sage"},
            "Confidence": {"Low": "Vetiver", "High": "Cardamom"}
        }

    def generate_affirmations(self, request: affirmation_request) -> affirmation_response:
        """Generate 12 affirmations based on quiz data and alignments, avoiding past themes."""
        
        # Calculate intensity tier from Q7 and Q8
        intensity_tier = self._calculate_intensity_tier(request.quizdata)
        
        # Make three LLM calls separately
        affirmation_prompt = self._create_affirmation_prompt(request)
        summary_prompt = self._create_quiz_summary_prompt(request)
        scent_prompt = self._create_scent_prompt(request, intensity_tier)
        
        # Call all three (could be parallelized in future if needed)
        affirmation_data = self._get_openai_response(affirmation_prompt, model=self.reasoning_model)
        summary_data = self._get_openai_response(summary_prompt, model="gpt-4o-mini")
        scent_data = self._get_openai_response(scent_prompt, model="gpt-4o-mini")
        
        if not affirmation_data:
            raise ValueError("Failed to generate affirmations")
        
        if not summary_data:
            raise ValueError("Failed to generate quiz summary")
        
        if not scent_data:
            raise ValueError("Failed to generate scent recommendations")
        
        affirmations = affirmation_data.get("affirmation", [])
        theme = affirmation_data.get("affirmation_theme", "")
        summary = summary_data.get("short_summary_of_quiz", "")
        base_scent = scent_data.get("base_scent", [])
        tertiary_scent = scent_data.get("tertiary_scent", [])
        
        if len(affirmations) != 12:
            raise ValueError(f"Expected 12 affirmations, got {len(affirmations)}")
        
        return affirmation_response(
            affirmation=affirmations, 
            affirmation_theme=theme,
            short_summary_of_quiz=summary,
            base_scent=base_scent,
            tertiary_scent=tertiary_scent
        )
    
    def _calculate_intensity_tier(self, quizdata: list) -> str:
        """Calculate intensity tier from Q7 (Restoration) and Q8 (Emotional Pattern) weights."""
        q7_weight = None
        q8_weight = None
        
        # Extract weights from quiz data
        for item in quizdata:
            question_lower = item.question.lower()
            
            # Q7: Restoration question (look for keywords)
            if "restore" in question_lower or "recharge" in question_lower or "energy" in question_lower:
                if item.answers:
                    # Calculate average weight from all selected answers
                    weights = [self.q7_weights.get(answer, 0) for answer in item.answers if answer in self.q7_weights]
                    if weights:
                        q7_weight = sum(weights) / len(weights)
            
            # Q8: Emotional Pattern question (look for keywords)
            if "feel" in question_lower or "emotional" in question_lower or "emotion" in question_lower:
                if item.answers:
                    # Calculate average weight from all selected answers
                    weights = [self.q8_weights.get(answer, 0) for answer in item.answers if answer in self.q8_weights]
                    if weights:
                        q8_weight = sum(weights) / len(weights)
        
        # Calculate intensity from average of Q7 and Q8
        if q7_weight is not None and q8_weight is not None:
            intensity = (q7_weight + q8_weight) / 2
        elif q7_weight is not None:
            intensity = q7_weight
        elif q8_weight is not None:
            intensity = q8_weight
        else:
            # Default to Mid if no matching questions found
            return "Mid"
        
        # Determine tier
        if intensity <= 35:
            return "Low"
        elif 40 <= intensity <= 60:
            return "Mid"
        elif intensity >= 65:
            return "High"
        else:
            return "Mid"

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

IMPORTANT: These preferences are MANDATORY when provided. You MUST incorporate them meaningfully into the affirmations, not just mention them superficially.

8. ENERGY CONTEXT: The user's restoration and emotional pattern responses (Q7 & Q8) are provided as secondary emotional context only. Do NOT let these dominate the tone or theme of the affirmations. They should subtly inform the emotional undertone, but the primary focus should remain on the user's goals, alignments, and preferences."""

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

    def _create_scent_prompt(self, request: affirmation_request, intensity_tier: str) -> str:
        """Build the prompt for scent selection (separate LLM call)."""
        
        system_prompt = """You are an expert scent curator who selects personalized fragrances based on user preferences and personality alignments.

Your task is to identify the base scent from the user's goal preferences, and add ONE secondary scent modifier based on intensity tier.

You MUST return valid JSON matching this exact structure:
{
  "base_scent": ["Sweet Orange", "Lemon", "Jasmine"] or "Lavender",
  "tertiary_scent": ["Rose", "Vanilla"] or "Rose"
}

CRITICAL STEP-BY-STEP PROCESS:

1. FIND USER'S GOAL(S):
   - Look at quiz_data for the goal-related question (usually "What Is Your Goal?")
   - Extract ALL goal answers the user selected
   - Example: User selected ["Joy & Happiness", "Self-Worth & Acceptance"]

2. MATCH GOALS TO SCENTS:
   - For EACH goal the user selected, find it in base_scent_info
   - Extract the "value" field (NOT the "goal" field) from base_scent_info
   - Combine all values into one list or string
   - Example: 
     * "Joy & Happiness" in base_scent_info has value: ["Sweet Orange", "Lemon", "Jasmine"]
     * "Self-Worth & Acceptance" in base_scent_info has value: ["Patchouli", "Neroli", "Cedarwood"]
     * Combined base_scent: ["Sweet Orange", "Lemon", "Jasmine", "Patchouli", "Neroli", "Cedarwood"]

3. IMPORTANT - RETURN THE SCENT NAMES, NOT GOAL NAMES:
   ❌ WRONG: {"base_scent": ["Joy & Happiness", "Self-Worth & Acceptance"]}
   ✅ CORRECT: {"base_scent": ["Sweet Orange", "Lemon", "Jasmine", "Patchouli", "Neroli", "Cedarwood"]}

4. SELECT SECONDARY SCENT (tertiary_scent field):
   CRITICAL RULES:
   - You are provided with intensity_tier (Low, Mid, or High) calculated from Q7 & Q8 weights
   - You are provided with goal_modifiers mapping each goal to Low/High modifier scents
   
   IF intensity_tier is "Low":
     - Add ONLY ONE modifier from the Low column for the user's goal(s)
     - Example: If user's goal is "Focus" and tier is Low → add ["Frankincense"]
   
   IF intensity_tier is "Mid":
     - DO NOT add any modifier
     - Return empty list [] or select one complementary scent based on personality
   
   IF intensity_tier is "High":
     - Add ONLY ONE modifier from the High column for the user's goal(s)
     - Example: If user's goal is "Focus" and tier is High → add ["Peppermint"]
   
   IMPORTANT:
   - Do NOT replace base oils
   - Add ONLY one modifier scent
   - DO NOT repeat any scents from base_scent
   - If user has multiple goals, choose the modifier for their primary goal

REAL EXAMPLE WITH ACTUAL DATA STRUCTURE:

base_scent_info provided:
[
  {"goal": "Joy & Happiness", "value": ["Sweet Orange", "Lemon", "Jasmine"]},
  {"goal": "Self-Worth & Acceptance", "value": ["Patchouli", "Neroli", "Cedarwood"]},
  {"goal": "Inner Peace & Balance", "value": ["Lavender", "Clary Sage", "Sandalwood"]}
]

User's quiz answer: "What Is Your Goal?" → ["Joy & Happiness", "Self-Worth & Acceptance"]

Correct response:
{
  "base_scent": ["Sweet Orange", "Lemon", "Jasmine", "Patchouli", "Neroli", "Cedarwood"],
  "tertiary_scent": ["Vanilla", "Rose"]
}

CRITICAL REMINDERS:
- base_scent MUST contain the actual scent names from the "value" field
- NEVER return goal names in base_scent
- If user has multiple goals, combine all their scent values
- Secondary scent (tertiary_scent) MUST follow the intensity_tier and goal_modifiers rules
- Use FULL Q7/Q8 weights (already calculated and provided as intensity_tier)
- Add ONLY one modifier, do not replace base oils"""

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

        # Convert base_scent_info to dict format
        scent_mapping = []
        for scent_item in request.base_scent_info:
            scent_mapping.append({
                "goal": scent_item.goal,
                "value": scent_item.value
            })

        user_payload = {
            "quiz_data": quiz_summary,
            "synergies": request.synergies or {},
            "harmonies": request.harmonies or {},
            "resonances": request.resonances or {},
            "polarities": request.polarities or {},
            "base_scent_info": scent_mapping,
            "intensity_tier": intensity_tier,
            "goal_modifiers": self.goal_modifiers,
            "instructions": "CRITICAL: 1) Find user's goal(s) from quiz_data, match each goal to base_scent_info, extract the VALUE field (the scent names, NOT the goal names), combine all values for base_scent. 2) Use intensity_tier and goal_modifiers to select ONE secondary scent modifier. If Low tier, add Low modifier for their goal. If High tier, add High modifier. If Mid tier, add no modifier or one complementary scent."
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
