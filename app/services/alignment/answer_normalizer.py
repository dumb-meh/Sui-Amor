"""Normalizes quiz answers to Answer_IDs for matching."""
from typing import Any, Dict, List, Optional


class AnswerNormalizer:
    """Maps quiz answer texts to Answer_IDs using fuzzy matching."""
    
    def __init__(self, answers_db: Dict[str, Dict[str, Any]]):
        """
        Initialize normalizer with answer database.
        
        Args:
            answers_db: Dictionary of answer_id -> answer data from data_store
        """
        self.answers_db = answers_db
        self._build_lookup_map()
    
    def _build_lookup_map(self) -> None:
        """Build lookup map for fast text -> Answer_ID matching."""
        self.text_to_id: Dict[str, str] = {}
        
        for answer_id, answer_data in self.answers_db.items():
            text = answer_data["text"]
            
            # Add exact match (case-insensitive)
            self.text_to_id[text.lower().strip()] = answer_id
            
            # Add variations
            # Remove special characters for fuzzy matching
            clean_text = self._clean_text(text)
            if clean_text != text.lower().strip():
                self.text_to_id[clean_text] = answer_id
    
    def _clean_text(self, text: str) -> str:
        """Clean text for fuzzy matching."""
        text = text.lower().strip()
        # Remove special chars but keep spaces
        text = ''.join(c if c.isalnum() or c.isspace() else '' for c in text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text
    
    def normalize_answer(self, answer_text: str) -> Optional[str]:
        """
        Map answer text to Answer_ID.
        
        Args:
            answer_text: Text from quiz JSON (e.g., "Blue", "I make it happen")
            
        Returns:
            Answer_ID (e.g., "COLOR_BLUE") or None if not found
        """
        # Try exact match first
        exact_key = answer_text.lower().strip()
        if exact_key in self.text_to_id:
            return self.text_to_id[exact_key]
        
        # Try cleaned version
        clean_key = self._clean_text(answer_text)
        if clean_key in self.text_to_id:
            return self.text_to_id[clean_key]
        
        # Try partial matching (for sub-options with long text)
        for lookup_text, answer_id in self.text_to_id.items():
            if clean_key in lookup_text or lookup_text in clean_key:
                # If there's significant overlap, consider it a match
                if len(clean_key) > 3:  # Avoid short false positives
                    return answer_id
        
        return None
    
    def normalize_quiz_answers(self, quiz_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize all answers from quiz JSON to Answer_IDs with selection order.
        
        Args:
            quiz_answers: List of quiz items from QuizEvaluationRequest
            
        Returns:
            List of {answer_id, selection_order, axes, category, text}
        """
        normalized = []
        global_order = 0  # Track overall selection order across all questions
        
        for quiz_item in quiz_answers:
            # Handle direct answers
            if "answers" in quiz_item and quiz_item["answers"]:
                for idx, answer_text in enumerate(quiz_item["answers"]):
                    answer_id = self.normalize_answer(answer_text)
                    
                    if answer_id and answer_id in self.answers_db:
                        answer_data = self.answers_db[answer_id]
                        normalized.append({
                            "answer_id": answer_id,
                            "selection_order": global_order,
                            "question_order": idx,  # Order within this question
                            "axes": answer_data["axes"],
                            "category": answer_data["category"],
                            "text": answer_data["text"],
                            "question": quiz_item.get("question", "")
                        })
                        global_order += 1
            
            # Handle sub_questions (hierarchical answers)
            if "sub_questions" in quiz_item and quiz_item["sub_questions"]:
                for sub_q in quiz_item["sub_questions"]:
                    if "sub_answers" in sub_q and sub_q["sub_answers"]:
                        for idx, sub_answer_text in enumerate(sub_q["sub_answers"]):
                            answer_id = self.normalize_answer(sub_answer_text)
                            
                            if answer_id and answer_id in self.answers_db:
                                answer_data = self.answers_db[answer_id]
                                normalized.append({
                                    "answer_id": answer_id,
                                    "selection_order": global_order,
                                    "question_order": idx,
                                    "axes": answer_data["axes"],
                                    "category": answer_data["category"],
                                    "text": answer_data["text"],
                                    "question": quiz_item.get("question", ""),
                                    "sub_question": sub_q.get("sub_question", "")
                                })
                                global_order += 1
        
        return normalized
    
    def get_unmatched_answers(self, quiz_answers: List[Dict[str, Any]]) -> List[str]:
        """
        Find answer texts that couldn't be matched to Answer_IDs.
        Useful for debugging.
        
        Args:
            quiz_answers: List of quiz items
            
        Returns:
            List of unmatched answer texts
        """
        unmatched = []
        
        for quiz_item in quiz_answers:
            if "answers" in quiz_item and quiz_item["answers"]:
                for answer_text in quiz_item["answers"]:
                    if not self.normalize_answer(answer_text):
                        unmatched.append(answer_text)
            
            if "sub_questions" in quiz_item and quiz_item["sub_questions"]:
                for sub_q in quiz_item["sub_questions"]:
                    if "sub_answers" in sub_q and sub_q["sub_answers"]:
                        for sub_answer_text in sub_q["sub_answers"]:
                            if not self.normalize_answer(sub_answer_text):
                                unmatched.append(sub_answer_text)
        
        return unmatched
