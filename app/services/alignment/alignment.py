"""Core alignment matching engine with 3-tier deterministic approach."""
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .answer_normalizer import AnswerNormalizer
from .data_store import AlignmentDataStore


class Alignment:
    """
    Three-tier alignment matcher:
    - Tier 1: Exact component match (if user selected exact alignment components)
    - Tier 2: Axis distance match (deterministic mathematical matching)
    - Tier 3: Vector similarity fallback (semantic search for novel combinations)
    """
    
    def __init__(self, csv_path: Optional[str] = None):
        """
        Initialize alignment matcher.
        
        Args:
            csv_path: Path to alignments CSV. Defaults to data/alignments.csv
        """
        # Initialize data store
        if csv_path is None:
            csv_path = Path(__file__).parent / "data" / "alignments.csv"
        
        print(f"[DEBUG] Initializing Alignment with CSV: {csv_path}")
        
        try:
            self.data_store = AlignmentDataStore(csv_path=csv_path)
            
            if self.data_store.answers:
                print(f"[DEBUG] Data store loaded: {len(self.data_store.answers)} answers, {len(self.data_store.alignments)} alignments")
            else:
                print(f"[WARNING] Data store initialized but no data loaded")
                
        except Exception as e:
            print(f"[ERROR] Failed to load data store: {e}")
            raise
        
        try:
            self.normalizer = AnswerNormalizer(self.data_store.answers)
            if self.data_store.answers:
                print(f"[DEBUG] Normalizer initialized with {len(self.normalizer.text_to_id)} text mappings")
        except Exception as e:
            print(f"[ERROR] Failed to initialize normalizer: {e}")
            raise
        
        # Matching configuration
        self.axis_distance_threshold = 3.0  # Max distance for Tier 2 matches
        self.min_results = 3  # Minimum results to return
        self.max_results = 12  # Maximum results to return
        
        print(f"[INFO] Alignment service initialized successfully")
    
    def match(self, request: Any, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Match quiz answers to alignments using 3-tier approach.
        
        Args:
            request: QuizEvaluationRequest with answers
            max_results: Maximum number of results to return
            
        Returns:
            List of alignment dictionaries sorted by relevance
        """
        if max_results is None:
            max_results = self.max_results
        
        # Normalize quiz answers to Answer_IDs
        if hasattr(request, 'answers'):
            quiz_answers = [answer.model_dump() for answer in request.answers]
        else:
            quiz_answers = request.get("answers", [])
        
        normalized_answers = self.normalizer.normalize_quiz_answers(quiz_answers)
        
        # Debug logging
        print(f"[DEBUG] Normalized {len(normalized_answers)} answers")
        print(f"[DEBUG] Data store has {len(self.data_store.answers)} answers, {len(self.data_store.alignments)} alignments")
        if normalized_answers:
            print(f"[DEBUG] First normalized: {normalized_answers[0]['answer_id']}")
        else:
            print("[WARNING] No answers were normalized!")
            # Check unmatched
            unmatched = self.normalizer.get_unmatched_answers(quiz_answers)
            if unmatched:
                print(f"[WARNING] Unmatched answers: {unmatched[:5]}")
        
        if not normalized_answers:
            return []
        
        # Compute user's axis profile
        user_profile = self._calculate_user_profile(normalized_answers)
        user_answer_ids = {ans["answer_id"] for ans in normalized_answers}
        user_categories = {ans["category"] for ans in normalized_answers}
        
        # TIER 1: Exact component match
        tier1_results = self._tier1_exact_match(
            user_answer_ids=user_answer_ids,
            normalized_answers=normalized_answers
        )
        
        if tier1_results:
            return self._format_results(tier1_results[:max_results], "exact")
        
        # TIER 2: Axis distance match (deterministic)
        tier2_results = self._tier2_axis_match(
            user_profile=user_profile,
            user_categories=user_categories
        )
        
        if tier2_results and tier2_results[0]["distance"] < self.axis_distance_threshold:
            return self._format_results(tier2_results[:max_results], "axis")
        
        # TIER 3: Vector similarity fallback (always returns results)
        tier3_results = self._tier3_vector_fallback(
            normalized_answers=normalized_answers,
            user_categories=user_categories,
            n_results=max_results
        )
        
        return self._format_results(tier3_results[:max_results], "vector")
    
    def _tier1_exact_match(
        self, 
        user_answer_ids: set, 
        normalized_answers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Tier 1: Find alignments whose components exactly match user's answers.
        Order-sensitive for SYNERGY/HARMONY types.
        """
        exact_matches = []
        
        # Create ordered sequence of answer IDs
        user_sequence = [ans["answer_id"] for ans in sorted(
            normalized_answers, 
            key=lambda x: x["selection_order"]
        )]
        
        for alignment_id, alignment in self.data_store.alignments.items():
            components = alignment["components"]
            components_set = set(components)
            
            # Check if all components are present in user's answers
            if not components_set.issubset(user_answer_ids):
                continue
            
            # For order-sensitive alignments, check sequence
            if alignment["component_order_matters"]:
                # Check if components appear in same order in user sequence
                if self._is_subsequence(components, user_sequence):
                    exact_matches.append({
                        "alignment": alignment,
                        "distance": 0.0,
                        "match_type": "exact_ordered"
                    })
            else:
                # Order doesn't matter, just set match
                exact_matches.append({
                    "alignment": alignment,
                    "distance": 0.0,
                    "match_type": "exact_unordered"
                })
        
        # Sort by number of components (more specific alignments first)
        exact_matches.sort(key=lambda x: len(x["alignment"]["components"]), reverse=True)
        
        return exact_matches
    
    def _tier2_axis_match(
        self,
        user_profile: Dict[str, float],
        user_categories: set
    ) -> List[Dict[str, Any]]:
        """
        Tier 2: Find alignments with closest axis distance (deterministic).
        Filters by category overlap.
        """
        candidates = []
        
        for alignment_id, alignment in self.data_store.alignments.items():
            # Filter by category (hard constraint)
            alignment_categories = set(alignment["categories"])
            if not alignment_categories.intersection(user_categories):
                continue
            
            # Calculate Euclidean distance in 5D axis space
            distance = self._euclidean_distance(user_profile, alignment["axes"])
            
            candidates.append({
                "alignment": alignment,
                "distance": distance,
                "match_type": "axis_distance"
            })
        
        # Sort by distance (closest first)
        candidates.sort(key=lambda x: x["distance"])
        
        return candidates
    
    def _tier3_vector_fallback(
        self,
        normalized_answers: List[Dict[str, Any]],
        user_categories: set,
        n_results: int
    ) -> List[Dict[str, Any]]:
        """
        Tier 3: Vector similarity fallback for completely novel combinations.
        Always returns results.
        """
        # Create query from user's answer texts
        query_parts = [ans["text"] for ans in normalized_answers]
        query_text = " ".join(query_parts)
        
        # Query ChromaDB (no category filter - we want results no matter what)
        alignment_ids = self.data_store.query_vector(
            query_text=query_text,
            n_results=n_results * 2  # Get more to filter
        )
        
        # Convert to result format
        results = []
        for alignment_id in alignment_ids:
            if alignment_id in self.data_store.alignments:
                alignment = self.data_store.alignments[alignment_id]
                results.append({
                    "alignment": alignment,
                    "distance": 999.0,  # Placeholder (ChromaDB doesn't return distance easily)
                    "match_type": "vector_similarity"
                })
        
        # If still no results, return any alignments from user's categories
        if not results:
            for alignment_id, alignment in self.data_store.alignments.items():
                alignment_categories = set(alignment["categories"])
                if alignment_categories.intersection(user_categories):
                    results.append({
                        "alignment": alignment,
                        "distance": 999.0,
                        "match_type": "category_fallback"
                    })
                if len(results) >= n_results:
                    break
        
        return results[:n_results]
    
    def _calculate_user_profile(self, normalized_answers: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate user's axis profile as weighted average.
        Earlier selections get higher weight (1.0, 0.5, 0.33, 0.25, ...)
        """
        if not normalized_answers:
            return {"energy": 0, "pace": 0, "orientation": 0, "structure": 0, "expression": 0}
        
        weighted_axes = {"energy": 0.0, "pace": 0.0, "orientation": 0.0, "structure": 0.0, "expression": 0.0}
        total_weight = 0.0
        
        # Sort by selection order
        sorted_answers = sorted(normalized_answers, key=lambda x: x["selection_order"])
        
        for idx, answer in enumerate(sorted_answers):
            # Weight decreases with selection order
            weight = 1.0 / (idx + 1)
            axes = answer["axes"]
            
            for axis in weighted_axes.keys():
                weighted_axes[axis] += axes[axis] * weight
            
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for axis in weighted_axes.keys():
                weighted_axes[axis] /= total_weight
        
        return weighted_axes
    
    def _euclidean_distance(self, profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two axis profiles."""
        distance_sq = 0.0
        for axis in ["energy", "pace", "orientation", "structure", "expression"]:
            diff = profile1.get(axis, 0) - profile2.get(axis, 0)
            distance_sq += diff * diff
        
        return math.sqrt(distance_sq)
    
    def _is_subsequence(self, components: List[str], user_sequence: List[str]) -> bool:
        """
        Check if components appear as subsequence in user_sequence.
        E.g., ["COLOR_RED", "COLOR_BLACK"] matches ["COLOR_RED", "NAV_FLOW", "COLOR_BLACK"]
        """
        if not components:
            return True
        
        comp_idx = 0
        for answer_id in user_sequence:
            if answer_id == components[comp_idx]:
                comp_idx += 1
                if comp_idx == len(components):
                    return True
        
        return False
    
    def _format_results(self, raw_results: List[Dict[str, Any]], tier: str) -> List[Dict[str, Any]]:
        """
        Format results for quiz_evaluation service.
        
        Returns format expected by quiz_evaluation:
        {
            "id": "HARMONY_RED_BLUE",
            "type": "HARMONY",
            "title": "Red and Blue – Harmony of Passion and Peace",
            "description": "When Red's intensity meets...",
            "match_tier": "exact",
            "confidence": 0.95
        }
        """
        formatted = []
        
        for result in raw_results:
            alignment = result["alignment"]
            distance = result.get("distance", 0.0)
            
            # Calculate confidence based on tier and distance
            if tier == "exact":
                confidence = 1.0
            elif tier == "axis":
                # Confidence decreases with distance (0 distance = 1.0 confidence)
                # Max distance of 3.0 = 0.5 confidence
                confidence = max(0.5, 1.0 - (distance / 6.0))
            else:  # vector fallback
                confidence = 0.5
            
            formatted.append({
                "id": alignment["id"],
                "type": alignment["type"],
                "title": alignment["title"],
                "description": alignment["description"],
                "match_tier": tier,
                "confidence": round(confidence, 2),
                "distance": round(distance, 3)
            })
        
        return formatted
    
    def reload_data(self) -> Dict[str, Any]:
        """
        Reload data from CSV (hot-reload after upload).
        
        Returns:
            Stats about reloaded data
        """
        stats = self.data_store.reload_from_csv()
        # Rebuild normalizer with new data
        self.normalizer = AnswerNormalizer(self.data_store.answers)
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded data."""
        return self.data_store.get_stats()
