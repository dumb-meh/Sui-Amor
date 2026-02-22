import json
import os
from typing import Any, Dict, List

import openai
from dotenv import load_dotenv

from app.services.alignment.alignment import Alignment

from .quiz_evaluation_schema import QuizEvaluationRequest, QuizEvaluationResponse

load_dotenv()


class QuizEvaluation:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.alignment_matcher = Alignment()
        self.max_results = 12
        self.tag_model = "gpt-4o-mini"

    def quiz_evaluation(self, request: QuizEvaluationRequest) -> QuizEvaluationResponse:
        """
        Evaluate quiz and return alignments grouped by type + AI-generated profile tags.
        Now uses deterministic alignment matching instead of vector search + LLM reasoning.
        """
        # Get deterministic alignment matches (no AI involved)
        matches = self.alignment_matcher.match(request, max_results=self.max_results)
        
        # Debug logging
        print(f"[DEBUG] Got {len(matches)} matches from alignment matcher")
        if matches:
            print(f"[DEBUG] First match: {matches[0].get('id', 'unknown')}")
        
        if not matches:
            # Return empty response if no matches
            print("[WARNING] No alignment matches found - returning empty response")
            return QuizEvaluationResponse(
                synergies={"items": []},
                harmonies={"items": []},
                resonances={"items": []},
                polarities={"items": []},
                profile_tags=[]
            )
        
        # Group matches by type (deterministic - no AI)
        synergies = [self._format_item(m) for m in matches if m["type"] == "SYNERGY"][:3]
        harmonies = [self._format_item(m) for m in matches if m["type"] == "HARMONY"][:3]
        resonances = [self._format_item(m) for m in matches if m["type"] == "RESONANCE"][:3]
        polarities = [self._format_item(m) for m in matches if m["type"] == "POLARITY"][:3]
        solos = [self._format_item(m) for m in matches if m["type"] == "SOLO"][:3]
        
        # If no specific types, put everything in synergies
        if not (synergies or harmonies or resonances or polarities):
            synergies = [self._format_item(m) for m in matches[:3]]
        
        # Generate profile tags with AI (only creative part)
        profile_tags = self._generate_profile_tags(request.answers, matches[:5])
        
        return QuizEvaluationResponse(
            synergies={"items": synergies},
            harmonies={"items": harmonies},
            resonances={"items": resonances},
            polarities={"items": polarities},
            profile_tags=profile_tags
        )
    
    def _format_item(self, match: Dict[str, Any]) -> Dict[str, str]:
        """Format match for response."""
        return {
            "id": match["id"],
            "title": match["title"],
            "description": match["description"]
        }
    
    def _generate_profile_tags(self, quiz_answers: List[Any], top_alignments: List[Dict[str, Any]]) -> List[str]:
        """
        Use AI to generate creative profile tags based on quiz answers and matched alignments.
        This is the only AI-powered part of the system.
        """
        if not quiz_answers:
            return []
        
        # Prepare data for AI
        quiz_data = [answer.model_dump() for answer in quiz_answers]
        alignment_summary = [
            {"name": a["title"], "type": a["type"]} 
            for a in top_alignments
        ]
        
        system_prompt = """You are an expert at creating personality tags based on quiz responses and alignment matches.

Generate 10 concise lowercase tags that capture the user's essence, energy, and personality.

Tags should be:
- Single words or short phrases (1-3 words)
- Lowercase
- Descriptive of personality, energy, or themes
- Based on both quiz answers and matched alignments

Examples: "driven", "calm", "creative", "high-energy", "reflective", "water", "nature-lover", "urban", "contemplative"

Return JSON format: {"tags": ["tag1", "tag2", ...]}"""
        
        user_payload = {
            "quiz_answers": quiz_data,
            "top_alignments": alignment_summary,
            "instruction": "Generate 10 unique lowercase tags that describe this person's energy, personality, and themes."
        }
        
        try:
            response = self.client.chat.completions.create(
                model=self.tag_model,
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
                ]
            )
            
            content = response.choices[0].message.content
            if content:
                result = json.loads(content)
                tags = result.get("tags", [])
                # Ensure tags are strings and lowercase
                return [str(tag).lower().strip() for tag in tags if str(tag).strip()][:10]
        except Exception as e:
            # Fallback: return empty tags if AI fails
            return []
        
        return []

