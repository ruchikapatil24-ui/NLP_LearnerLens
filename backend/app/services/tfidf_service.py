from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf(
    student_text: str,
    reference_text: str = "",
    corpus: List[str] = None
) -> Dict[str, Any]:
    """
    Practical Point 6:
    Calculate Term Frequency–Inverse Document Frequency using scikit-learn's TfidfVectorizer.
    Extracts top terms, computes cosine similarity against reference document,
    and returns a matrix summary for heatmap visualization.
    """
    if not student_text.strip():
        return {
            "top_terms": [],
            "reference_similarity": 0.0,
            "vocabulary_size": 0,
            "matrix_heatmap": [],
            "formula_explanation": "TF-IDF(t,d) = TF(t,d) * IDF(t)",
            "interpretation": "Empty input text provided."
        }

    # Assemble document corpus
    documents = [student_text]
    has_ref = bool(reference_text and reference_text.strip())
    if has_ref:
        documents.append(reference_text)

    if corpus:
        for c in corpus:
            if c and c.strip() and c not in documents:
                documents.append(c)

    # If only 1 document is present, add synthetic academic reference corpus for IDF variation
    if len(documents) < 3:
        documents.extend([
            "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plant chloroplasts.",
            "Newton's laws govern classical mechanics, force, mass, momentum, action and reaction between interacting bodies.",
            "Algorithms use recursion, iteration, sorting, and binary search to solve computational problems with time complexity."
        ])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=1,
        ngram_range=(1, 1),
        sublinear_tf=True
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = np.array(vectorizer.get_feature_names_out())
    except Exception as e:
        return {
            "top_terms": [],
            "reference_similarity": 0.0,
            "vocabulary_size": 0,
            "matrix_heatmap": [],
            "formula_explanation": "TF-IDF(t,d) = TF(t,d) * IDF(t)",
            "interpretation": f"Vectorization error: {str(e)}"
        }

    # Student document is always index 0
    student_vec = tfidf_matrix[0].toarray().flatten()
    sorted_indices = np.argsort(student_vec)[::-1]

    ref_words_lower = set(reference_text.lower().split()) if has_ref else set()

    top_terms = []
    for idx in sorted_indices:
        score = float(student_vec[idx])
        if score > 0.001:
            term = str(feature_names[idx])
            top_terms.append({
                "term": term,
                "score": round(score, 4),
                "in_reference": (term in ref_words_lower) or any(term in rw for rw in ref_words_lower)
            })
        if len(top_terms) >= 12:
            break

    # Cosine similarity between student response and reference answer
    ref_sim = 0.0
    if has_ref:
        ref_vec = tfidf_matrix[1].toarray()
        cos_sim = cosine_similarity(student_vec.reshape(1, -1), ref_vec)[0][0]
        ref_sim = round(float(max(0.0, min(1.0, cos_sim))), 4)

    # Build heatmap rows for top 8 terms across student and reference
    sample_indices = [idx for idx in sorted_indices if student_vec[idx] > 0.001][:8]
    matrix_heatmap = []
    for idx in sample_indices:
        term = str(feature_names[idx])
        s_score = round(float(student_vec[idx]), 3)
        r_score = round(float(tfidf_matrix[1].toarray()[0][idx]), 3) if has_ref else 0.0
        matrix_heatmap.append({
            "term": term,
            "student_tfidf": s_score,
            "reference_tfidf": r_score,
            "idf_weight": round(float(vectorizer.idf_[idx]), 3)
        })

    formula = (
        "TF-IDF(t, d) = TF(t, d) * IDF(t)  where IDF(t) = ln((1 + n) / (1 + df(t))) + 1\n"
        "• TF(t,d): Relative term frequency in student answer\n"
        "• IDF(t): Inverse document frequency reflecting uniqueness across the corpus"
    )

    interpretation = (
        f"Extracted {len(top_terms)} key lexical terms. Vocabulary size across corpus: {len(feature_names)}. "
        + (f"Cosine similarity with reference answer: {round(ref_sim * 100, 1)}%. " if has_ref else "")
        + "Terms with high TF-IDF are unique domain identifiers that distinguish this answer from generic writing. "
        "High overlap with reference terms indicates accurate subject terminology."
    )

    return {
        "top_terms": top_terms,
        "reference_similarity": ref_sim,
        "vocabulary_size": len(feature_names),
        "matrix_heatmap": matrix_heatmap,
        "formula_explanation": formula,
        "interpretation": interpretation
    }
