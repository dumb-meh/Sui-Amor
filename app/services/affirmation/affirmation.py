import json
import os
from typing import Any, Dict

import openai
from dotenv import load_dotenv

from .affirmation_schema import affirmation_request, affirmation_response
from app.utils.cache_manager import cache_manager

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
        
        # Resolve scent direction from the direction matrix (Q9+Q2+Q8+Q10 lookup)
        direction = self._resolve_direction_from_matrix(request.quizdata)
        
        # Make three LLM calls separately
        affirmation_prompt = self._create_affirmation_prompt(request)
        summary_prompt = self._create_quiz_summary_prompt(request)
        scent_prompt = self._create_scent_prompt(request, direction)
        
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
        
        response = affirmation_response(
            affirmation=affirmations, 
            affirmation_theme=theme,
            short_summary_of_quiz=summary,
            base_scent=base_scent,
            tertiary_scent=tertiary_scent
        )

        # Persist goal + religious preference to Redis (upsert — overwrites existing)
        try:
            goal = self._extract_goal(request.quizdata)
            cache_manager.save_user_goal(
                user_id=request.user_id,
                goal=goal,
                religious_preference=request.religious_or_spritual_preference,
            )
        except Exception as e:
            print(f"[WARN] Failed to save user goal to Redis: {e}")

        return response

    def _extract_goal(self, quizdata: list) -> str:
        """Extract the goal answer(s) from the 9th quiz question (index 8)."""
        try:
            ninth_question = quizdata[8]  # 0-indexed
            if ninth_question.answers:
                return ", ".join(ninth_question.answers)
        except (IndexError, AttributeError):
            pass
        return ""
    
    def _resolve_direction_from_matrix(self, quizdata: list) -> str:
        """
        Determine the scent direction (Calming / Neutral / Energizing) for a
        user by looking up their Q9 goal + Q2 identity + Q8 emotional state +
        Q10 obstacle combination in the uploaded direction matrix.

        Falls back to "Neutral" if the matrix has not been uploaded yet or
        the user's specific combination is not found.

        Returns one of: "Calming", "Neutral", "Energizing"
        """
        from app.utils.direction_matrix import resolve_direction_for_goals

        # ---- Extract Q9 goals -------------------------------------------------
        goals: list[str] = []
        for item in quizdata:
            q = item.question.lower()
            if "goal" in q or "what is your goal" in q:
                if item.answers:
                    goals.extend(item.answers)
                break

        if not goals:
            return "Neutral"

        # ---- Extract Q2 identity / navigation statement -----------------------
        # Q2 asks something like "Which quote feels most you?" or
        # "Which statement best describes how you navigate life?"
        q2_answer = ""
        for item in quizdata:
            q = item.question.lower()
            if "quote" in q or "feels most you" in q or "navigation" in q \
                    or ("statement" in q and "describe" in q):
                if item.answers:
                    q2_answer = item.answers[0]
                break

        # ---- Extract Q8 emotional state ----------------------------------------
        # Q8 asks something like "What Emotion Best Describes Your General State
        # of Being?" — deliberately avoid colour questions that contain "feel".
        q8_state = ""
        for item in quizdata:
            q = item.question.lower()
            # Must contain emotional/emotion AND be about a "state" or "being",
            # NOT a colour/visual question.
            is_emotional_state = (
                ("emotion" in q or "emotional" in q)
                and ("state" in q or "being" in q or "describes" in q or "pattern" in q)
            )
            # Exclude colour / sensory questions even if they happen to use "feel"
            is_color_question = "color" in q or "colour" in q or "drawn to" in q
            if is_emotional_state and not is_color_question:
                if item.answers:
                    q8_state = item.answers[0]
                break

        # ---- Extract Q10 obstacle ----------------------------------------------
        q10_obstacle = ""
        for item in quizdata:
            q = item.question.lower()
            if "obstacle" in q or "struggle" in q or "biggest" in q:
                if item.answers:
                    q10_obstacle = item.answers[0]
                break

        direction = resolve_direction_for_goals(
            goals=goals,
            q2_answer=q2_answer,
            q8_state=q8_state,
            q10_obstacle=q10_obstacle,
        )

        print(f"[DEBUG] Direction matrix result: {direction} "
              f"(goals={goals}, q2={q2_answer!r}, q8={q8_state!r}, q10={q10_obstacle!r})")

        return direction  # "Calming" | "Neutral" | "Energizing"

    def _create_affirmation_prompt(self, request: affirmation_request) -> str:
        """Build the prompt for generating affirmations, dynamically based on affirmation_type."""

        # Build preference context blocks
        religious_block = self._build_religious_instructions(request)
        astrology_block = self._build_astrology_instructions(request)

        preference_lines = []
        if religious_block:
            preference_lines.append(religious_block)
        if astrology_block:
            preference_lines.append(astrology_block)
        if request.holiday_preference:
            preference_lines.append(
                f"- HOLIDAY: The user celebrates {request.holiday_preference}. "
                f"Incorporate themes of {request.holiday_preference} such as celebration, renewal, gratitude, or community. "
                f"Use imagery and symbolism associated with {request.holiday_preference}."
            )

        preference_text = "\n".join(preference_lines) if preference_lines else "- No additional preference layers. Keep affirmations universal."

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

7. PREFERENCE LAYERS - APPLY THESE EXACTLY AS INSTRUCTED:
{preference_text}

IMPORTANT: Preference layers must be applied proportionally to their influence level. Do not override the user's core goal or alignment — preferences are additive layers, not replacements.

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

        # Add active preference fields to payload for AI reference
        if request.religious_or_spritual_preference:
            user_payload["religious_or_spiritual_preference"] = request.religious_or_spritual_preference
            user_payload["religious_influence_score"] = request.religious_preference_priority_score
        if request.astrology_preference:
            user_payload["astrology_preference"] = request.astrology_preference
        if request.holiday_preference:
            user_payload["holiday_preference"] = request.holiday_preference

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Preference instruction builders
    # ------------------------------------------------------------------

    # Themes and wording directions per broad category
    _RELIGIOUS_THEMES = {
        "abrahamic": {
            "label": "Abrahamic traditions (Christianity, Islam, Judaism, and related faiths)",
            "themes": ["faith", "perseverance", "grace", "redemption", "guidance", "purpose", "renewal"],
            "wording": (
                "Use language rooted in faith, divine guidance, and grace. "
                "Words like 'blessed', 'grace', 'purpose', 'surrender to guidance', and 'renewed by faith' "
                "are welcome. Avoid naming specific denominations."
            ),
        },
        "eastern": {
            "label": "Eastern traditions (Buddhism, Hinduism, Taoism, Sikhism, Jainism, and related paths)",
            "themes": ["mindfulness", "balance", "harmony", "compassion", "stillness", "detachment", "inner peace"],
            "wording": (
                "Use language rooted in inner stillness, non-attachment, and compassion. "
                "Words like 'present moment', 'flow', 'balance', 'harmony', 'release', and 'equanimity' "
                "are welcome. Keep it universal across Eastern traditions."
            ),
        },
        "spiritual": {
            "label": "Spiritual / Nature / Energy-based (meditation, ancestral, nature, energy work)",
            "themes": ["grounding", "intuition", "connectedness", "cycles", "growth", "energy", "reflection"],
            "wording": (
                "Use language rooted in natural cycles, energetic connection, and inner knowing. "
                "Words like 'rooted', 'aligned', 'energy', 'intuition', 'cycles of growth', and 'connected to all' "
                "are welcome. Keep the tone expansive and nature-aware."
            ),
        },
    }

    # Influence level descriptions for 1–5 score
    _INFLUENCE_LEVELS = {
        1: "Do NOT use any spiritual or religious language. Keep affirmations fully secular and universal.",
        2: "Apply only the lightest touch of spiritual language — one or two words at most across all 12 affirmations.",
        3: "Weave spiritual themes in moderately — present in roughly half the affirmations, but never overpowering the core message.",
        4: "Apply spiritual themes with strong presence — most affirmations should carry the wording direction and selected themes.",
        5: "Let spiritual themes define the tone — every affirmation should reflect the wording direction and associated themes of this tradition.",
    }

    # Zodiac sign traits
    _ZODIAC_TRAITS = {
        "aries":       ["confidence", "initiative", "courage", "ambition", "action"],
        "taurus":      ["stability", "patience", "loyalty", "grounding", "resilience"],
        "gemini":      ["curiosity", "adaptability", "communication", "creativity", "energy"],
        "cancer":      ["emotional depth", "protection", "intuition", "nurturing", "sensitivity"],
        "leo":         ["leadership", "confidence", "passion", "expression", "warmth"],
        "virgo":       ["discipline", "growth", "reflection", "organization", "improvement"],
        "libra":       ["balance", "harmony", "connection", "diplomacy", "beauty"],
        "scorpio":     ["transformation", "intensity", "determination", "depth", "resilience"],
        "sagittarius": ["exploration", "optimism", "freedom", "learning", "expansion"],
        "capricorn":   ["discipline", "ambition", "structure", "persistence", "responsibility"],
        "aquarius":    ["individuality", "innovation", "independence", "vision", "originality"],
        "pisces":      ["imagination", "compassion", "intuition", "healing", "emotional openness"],
    }

    def _build_religious_instructions(self, request: affirmation_request) -> str:
        """
        Build the religious/spiritual preference instruction block.

        Returns an empty string when:
        - No preference is set (None / empty)
        - The preference maps to no known category
        - The score is 1 (user explicitly wants no spiritual language)
        """
        pref = (request.religious_or_spritual_preference or "").strip().lower()
        if not pref:
            return ""

        # Resolve broad category from the raw preference string
        if pref in ("abrahamic", "abrahamic traditions"):
            category_key = "abrahamic"
        elif pref in ("eastern", "eastern traditions"):
            category_key = "eastern"
        elif pref in ("spiritual", "spiritual / nature / energy-based", "nature", "energy"):
            category_key = "spiritual"
        else:
            # Unknown category — skip silently
            return ""

        # Resolve influence score (default to 3 if not provided or unparseable)
        try:
            score = int(request.religious_preference_priority_score)
            score = max(1, min(5, score))  # clamp 1–5
        except (TypeError, ValueError):
            score = 3

        if score == 1:
            # Score 1 = user explicitly wants no spiritual influence at all
            return "- RELIGIOUS/SPIRITUAL: Keep affirmations fully secular. Do NOT use any spiritual or religious language whatsoever."

        cat = self._RELIGIOUS_THEMES[category_key]
        influence_instruction = self._INFLUENCE_LEVELS[score]
        themes_str = ", ".join(cat["themes"])

        return (
            f"- RELIGIOUS/SPIRITUAL ({cat['label']}, influence level {score}/5):\n"
            f"  Influence rule: {influence_instruction}\n"
            f"  Associated themes to draw from: {themes_str}\n"
            f"  Wording direction: {cat['wording']}"
        )

    def _build_astrology_instructions(self, request: affirmation_request) -> str:
        """
        Build the astrology preference instruction block.

        Astrology is treated as a LIGHT seasonal personalisation layer only.
        It must never dominate the affirmation tone or theme.
        Returns an empty string when no preference is set.
        """
        sign = (request.astrology_preference or "").strip().lower()
        if not sign:
            return ""

        traits = self._ZODIAC_TRAITS.get(sign)
        if not traits:
            return ""

        traits_str = ", ".join(traits)
        return (
            f"- ASTROLOGY ({sign.capitalize()} energy — light seasonal layer only):\n"
            f"  Subtly weave 1–2 of these traits into a small number of affirmations (not all 12): {traits_str}.\n"
            f"  This must feel like a whisper of personalisation, NOT the defining tone. "
            f"The user's goal and alignments remain the primary focus."
        )

    def _create_quiz_summary_prompt(self, request: affirmation_request) -> str:
        """Build the prompt for generating quiz summary (separate LLM call)."""
        
        system_prompt = """You are an expert at analyzing quiz responses and creating thoughtful, highly cohesive, and personalized summaries.

Your task is to generate a short summary of the user's quiz responses that reflects their values, patterns, and current focus. The summary MUST flow as a single evolving narrative where ideas connect and reinforce each other instead of resetting.

CRITICAL NARRATIVE INSTRUCTIONS:
1. Cohesion: Each section and paragraph MUST transition naturally into the next.
2. Continuity: Reference earlier points in subsequent sentences, rather than introducing new concepts in isolation.
3. Unified Voice: The entire summary should read as one continuous interpretation, seamlessly connected, NOT multiple segmented observations stacked together.

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
            "instructions": "Generate a highly cohesive and personalized quiz summary. Ensure smooth transitions between paragraphs and concepts so the entire summary flows seamlessly without sounding segmented. Follow the exact structure specified, using \\n for line breaks and - for bullet points."
        }

        return json.dumps({"system": system_prompt, "payload": user_payload}, ensure_ascii=False)

    def _create_scent_prompt(self, request: affirmation_request, direction: str) -> str:
        """
        Build the prompt for scent selection (separate LLM call).

        The direction (Calming / Neutral / Energizing) has already been
        resolved deterministically from the direction matrix.  We map it to the
        correct ScentDirections key here in Python, pre-resolve base_scent,
        and only ask the AI to select ONE complementary tertiary modifier.
        """
        from app.utils.direction_matrix import DIRECTION_TO_SCENT_KEY

        # Map direction → ScentDirections key
        scent_key = DIRECTION_TO_SCENT_KEY.get(direction, "Neutral")

        # ----------------------------------------------------------------
        # Pre-resolve base_scent entirely in Python — no AI guesswork.
        # ----------------------------------------------------------------
        base_scent: list[str] = []
        for scent_item in request.base_scent_info:
            directions_obj = scent_item.directions
            if scent_key == "Calming/Grounding":
                oils = directions_obj.Calming_Grounding
            elif scent_key == "Elevating/Energizing":
                oils = directions_obj.Elevating_Energizing
            else:
                oils = directions_obj.Neutral
            base_scent.extend(oils)

        # De-duplicate while preserving order
        seen: set[str] = set()
        unique_base: list[str] = []
        for oil in base_scent:
            if oil and oil.lower() not in seen:
                seen.add(oil.lower())
                unique_base.append(oil)

        system_prompt = f"""You are an expert scent curator for Sui Amor, a personal growth platform.

The user's base essential oil blend has already been selected based on their goal and emotional direction.
Your ONLY task is to recommend ONE optional tertiary (modifier) scent that complements this blend.

Direction: {direction} (scent profile: {scent_key})
Base blend (already selected): {unique_base}

You MUST return valid JSON matching this exact structure:
{{
  "base_scent": {json.dumps(unique_base)},
  "tertiary_scent": ["one modifier scent name"]
}}

RULES FOR TERTIARY SCENT:
- Choose exactly ONE modifier scent that complements the direction and base blend
- Direction "{direction}" guidance:
  * Calming    → choose a grounding/soothing modifier (e.g. Vetiver, Myrrh, Cedarwood, Sandalwood)
  * Neutral    → choose a balancing modifier (e.g. Rose, Geranium, Clary Sage, Palmarosa)
  * Energizing → choose an uplifting/stimulating modifier (e.g. Peppermint, Lemon, Ginger, Bergamot)
- Do NOT repeat any scent already in base_scent
- Do NOT replace or modify base_scent — return it exactly as provided
- Return ONLY the modifier name(s) in tertiary_scent, not goal names

IMPORTANT: The base_scent in your response MUST be exactly: {json.dumps(unique_base)}
Do not change it."""

        # Build quiz summary for context
        quiz_summary = []
        for item in request.quizdata:
            question_data = {"question": item.question}
            if item.answers:
                question_data["answers"] = item.answers
            if item.sub_questions:
                sub_q_data = [
                    {"sub_question": sq.sub_question, "sub_answers": sq.sub_answers}
                    for sq in item.sub_questions
                ]
                question_data["sub_questions"] = sub_q_data
            quiz_summary.append(question_data)

        user_payload = {
            "quiz_data": quiz_summary,
            "direction": direction,
            "scent_direction_key": scent_key,
            "base_scent_already_resolved": unique_base,
            "goal_modifiers": self.goal_modifiers,
            "instructions": (
                f"The base_scent is already resolved as {unique_base}. "
                f"Choose ONE tertiary modifier scent that fits the '{direction}' direction "
                f"and does not duplicate any oil in base_scent."
            ),
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
