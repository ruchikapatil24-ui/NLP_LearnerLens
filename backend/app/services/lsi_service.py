from typing import Dict, Any, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

def perform_lsi(
    student_text: str,
    reference_text: str = "",
    n_components: int = 2,
    corpus: List[str] = None
) -> Dict[str, Any]:
    """
    Practical Point 8:
    Latent Semantic Indexing using TfidfVectorizer + TruncatedSVD.
    Reduces document-term space into 2 latent semantic concepts,
    projects student and reference answers, and computes latent similarity.
    """
    if not student_text.strip():
        return {
            "explained_variance_ratio": [0.0, 0.0],
            "student_coordinates": [0.0, 0.0],
            "reference_coordinates": [0.0, 0.0],
            "similarity_score": 0.0,
            "latent_topics": [],
            "explanation": "No text provided for Latent Semantic Indexing.",
            "short_answer_warning": "Input is empty."
        }

    docs = [student_text]
    has_ref = bool(reference_text and reference_text.strip())
    if has_ref:
        docs.append(reference_text)

    if corpus:
        for c in corpus:
            if c and c.strip() and c not in docs:
                docs.append(c)

    # Add standard domain anchor documents if corpus is small
    if len(docs) < 4:
        docs.extend([
            "Photosynthesis converts light into chemical energy using chlorophyll and carbon dioxide.",
            "Newton's laws define classical mechanics, action-reaction forces, and momentum conservation.",
            "Recursion is a programming technique with self-calling functions and base termination cases.",
            "The Industrial Revolution replaced cottage manual labor with steam power and urban factories."
        ])

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    try:
        tfidf_mat = vectorizer.fit_transform(docs)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        return {
            "explained_variance_ratio": [0.0, 0.0],
            "student_coordinates": [0.0, 0.0],
            "reference_coordinates": [0.0, 0.0],
            "similarity_score": 0.0,
            "latent_topics": [],
            "explanation": f"Vectorization failed: {str(e)}",
            "short_answer_warning": "Vocabulary extraction failed."
        }

    n_comp = min(n_components, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
    if n_comp < 2:
        n_comp = 1

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    try:
        reduced_docs = svd.fit_transform(tfidf_mat)
        var_ratios = [round(float(v), 3) for v in svd.explained_variance_ratio_]
    except Exception:
        reduced_docs = np.zeros((len(docs), 2))
        var_ratios = [0.5, 0.5]

    # Student coords
    s_coords = [round(float(reduced_docs[0, i]), 3) for i in range(min(2, reduced_docs.shape[1]))]
    if len(s_coords) == 1:
        s_coords.append(0.0)

    # Reference coords
    r_coords = [0.0, 0.0]
    lsi_sim = 0.0
    if has_ref:
        r_coords = [round(float(reduced_docs[1, i]), 3) for i in range(min(2, reduced_docs.shape[1]))]
        if len(r_coords) == 1:
            r_coords.append(0.0)

        # Cosine similarity in latent space
        norm_s = np.linalg.norm(reduced_docs[0])
        norm_r = np.linalg.norm(reduced_docs[1])
        if norm_s > 0 and norm_r > 0:
            cos = np.dot(reduced_docs[0], reduced_docs[1]) / (norm_s * norm_r)
            lsi_sim = round(float(max(0.0, min(1.0, cos))), 4)
        else:
            lsi_sim = 0.0

    # Extract top terms contributing to latent topics
    latent_topics = []
    for comp_idx in range(min(2, len(svd.components_))):
        comp = svd.components_[comp_idx]
        top_idx = np.argsort(comp)[::-1][:5]
        top_words = [f"{feature_names[i]} ({round(float(comp[i]), 2)})" for i in top_idx]
        latent_topics.append({
            "topic_id": f"Dimension {comp_idx + 1}",
            "explained_variance": var_ratios[comp_idx] if comp_idx < len(var_ratios) else 0.0,
            "dominant_terms": top_words
        })

    words_count = len(student_text.split())
    short_warning = None
    if words_count < 10:
        short_warning = (
            f"Note: Response is brief ({words_count} words). Short answers create sparse vectors "
            "that may distort latent projections. LSI should be treated as one supporting signal rather than a definitive test."
        )

    explanation = (
        f"Truncated SVD reduced {tfidf_mat.shape[1]} vocabulary features into {n_comp} latent semantic dimensions. "
        f"Total explained variance by primary dimensions: {round(sum(var_ratios) * 100, 1)}%. "
        + (f"Latent Cosine Similarity to Reference: {round(lsi_sim * 100, 1)}%. " if has_ref else "")
        + "LSI captures conceptual relationships and synonomy even when the student uses different word choices than the reference."
    )

    return {
        "explained_variance_ratio": var_ratios,
        "student_coordinates": s_coords,
        "reference_coordinates": r_coords if has_ref else None,
        "similarity_score": lsi_sim,
        "latent_topics": latent_topics,
        "explanation": explanation,
        "short_answer_warning": short_warning
    }
