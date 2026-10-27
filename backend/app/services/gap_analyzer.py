import re
from typing import List, Dict, Any

COLLOQUIAL_PATTERNS = [
    r'\bstuff\b', r'\bthings\b', r'\bkind of\b', r'\bsort of\b',
    r'\blike if\b', r'\byou know\b', r'\bbig\b', r'\bbad\b'
]

def analyze_learning_gaps(
    student_response: str,
    reference_answer: str = "",
    tfidf_top_terms: List[Dict[str, Any]] = None,
    pos_distribution: List[Dict[str, Any]] = None,
    similarity_score: float = 0.0
) -> Dict[str, Any]:
    """
    Identifies specific areas where the answer may be incomplete, lacks technical vocabulary,
    or relies on colloquial approximations, and generates constructive pedagogical recommendations.
    """
    gaps: List[str] = []
    recommendations: List[str] = []
    triggers: List[str] = []

    clean_text = student_response.strip()
    words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', clean_text)]
    word_count = len(words)

    # 1. Answer Length & Completeness
    if word_count < 12:
        gaps.append("Answer is exceptionally brief and omits explanatory reasoning or scientific mechanisms.")
        recommendations.append("Expand answer by explaining 'why' and 'how' rather than stating a single isolated fact.")
        triggers.append(f"Word count threshold triggered: {word_count} words (< 12 target)")
    elif word_count < 22:
        gaps.append("Explanation covers only surface points without detailing underlying processes.")
        recommendations.append("Provide a concrete real-world example or intermediate step to demonstrate depth.")
        triggers.append(f"Moderate response length: {word_count} words")

    # 2. Colloquial vs Domain Vocabulary
    found_colloquial = []
    for pattern in COLLOQUIAL_PATTERNS:
        match = re.search(pattern, clean_text, re.IGNORECASE)
        if match:
            found_colloquial.append(match.group(0))

    if found_colloquial:
        colloquial_str = ", ".join(f"'{c}'" for c in found_colloquial[:3])
        gaps.append(f"Relies on informal or imprecise language (e.g. {colloquial_str}) instead of formal academic terminology.")
        recommendations.append(f"Replace informal terms like {colloquial_str} with precise domain vocabulary.")
        triggers.append(f"Identified colloquial terms: {colloquial_str}")

    # 3. Reference Term Overlap
    if reference_answer and reference_answer.strip():
        ref_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', reference_answer.lower()))
        common_stops = {"this", "that", "with", "from", "when", "where", "which", "their", "there", "about", "other"}
        content_ref_words = ref_words - common_stops
        student_set = set(words)
        overlap = content_ref_words.intersection(student_set)
        missing_terms = list(content_ref_words - student_set)[:4]

        if len(missing_terms) >= 3 and len(overlap) <= 1:
            missing_str = ", ".join(f"'{m}'" for m in missing_terms)
            gaps.append(f"Missing core reference concepts such as: {missing_str}.")
            recommendations.append(f"Review key conceptual elements: {missing_str}.")
            triggers.append(f"Low reference keyword recall ({len(overlap)} / {len(content_ref_words)})")

    # 4. Syntactic Complexity from POS
    noun_count = 0
    verb_count = 0
    adj_count = 0
    if pos_distribution:
        for cat in pos_distribution:
            c_name = cat.get("category")
            if c_name == "Noun":
                noun_count = cat.get("count", 0)
            elif c_name == "Verb":
                verb_count = cat.get("count", 0)
            elif c_name == "Adjective":
                adj_count = cat.get("count", 0)

    if word_count > 10 and adj_count == 0 and (noun_count / max(1, verb_count) < 1.0):
        gaps.append("Sentence structure lacks descriptive adjectives or noun modifiers to qualify scientific statements.")
        recommendations.append("Practice using qualifying terms and precise descriptors to add clarity.")
        triggers.append("Low descriptive adjective density detected.")

    # 5. Semantic Relevance
    if similarity_score > 0 and similarity_score < 0.35:
        gaps.append("Low semantic alignment with expected reference material; may indicate topic confusion.")
        recommendations.append("Revisit the chapter overview or ask the teacher for clarification on this topic.")
        triggers.append(f"Reference semantic similarity below threshold: {round(similarity_score, 2)}")

    # Default encouragement if no significant gaps
    if not gaps:
        gaps.append("Demonstrates solid conceptual grasp with appropriate vocabulary and structure.")
        recommendations.append("Try exploring a more advanced extension question or explaining subtle boundary cases.")
        triggers.append("Passed all baseline lexical and syntactic criteria")

    return {
        "learning_gaps": gaps,
        "recommendations": recommendations,
        "rule_triggers": triggers
    }
