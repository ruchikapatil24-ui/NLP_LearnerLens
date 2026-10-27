import re
import math
from typing import Dict, Any, List
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
import numpy as np
from gensim.models import Word2Vec

TAG_DESCRIPTIONS = {
    'CC': ('Coordinating conjunction', 'Conjunction'),
    'CD': ('Cardinal digit', 'Number'),
    'DT': ('Determiner', 'Determiner'),
    'EX': ('Existential there', 'Pronoun'),
    'FW': ('Foreign word', 'Other'),
    'IN': ('Preposition/subordinating conjunction', 'Preposition'),
    'JJ': ('Adjective (positive)', 'Adjective'),
    'JJR': ('Adjective (comparative)', 'Adjective'),
    'JJS': ('Adjective (superlative)', 'Adjective'),
    'LS': ('List marker', 'Other'),
    'MD': ('Modal auxiliary', 'Verb'),
    'NN': ('Noun, singular or mass', 'Noun'),
    'NNS': ('Noun, plural', 'Noun'),
    'NNP': ('Proper noun, singular', 'Noun'),
    'NNPS': ('Proper noun, plural', 'Noun'),
    'PDT': ('Predeterminer', 'Determiner'),
    'POS': ('Possessive ending', 'Other'),
    'PRP': ('Personal pronoun', 'Pronoun'),
    'PRP$': ('Possessive pronoun', 'Pronoun'),
    'RB': ('Adverb (positive)', 'Adverb'),
    'RBR': ('Adverb (comparative)', 'Adverb'),
    'RBS': ('Adverb (superlative)', 'Adverb'),
    'RP': ('Particle', 'Other'),
    'TO': ('to', 'Preposition'),
    'UH': ('Interjection', 'Other'),
    'VB': ('Verb, base form', 'Verb'),
    'VBD': ('Verb, past tense', 'Verb'),
    'VBG': ('Verb, gerund/present participle', 'Verb'),
    'VBN': ('Verb, past participle', 'Verb'),
    'VBP': ('Verb, non-3rd person singular present', 'Verb'),
    'VBZ': ('Verb, 3rd person singular present', 'Verb'),
    'WDT': ('Wh-determiner', 'Determiner'),
    'WP': ('Wh-pronoun', 'Pronoun'),
    'WP$': ('Possessive wh-pronoun', 'Pronoun'),
    'WRB': ('Wh-adverb', 'Adverb'),
}

def analyze_pos(text: str) -> Dict[str, Any]:
    """Execute POS tagging using NLTK and generate distribution metrics."""
    if not text.strip():
        return {
            "tokens_with_tags": [],
            "category_distribution": [],
            "explanation": "No text provided for POS analysis."
        }

    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    # Filter out empty or whitespace tokens
    clean_tokens = [t for t in tokens if t.strip()]
    if not clean_tokens:
        return {
            "tokens_with_tags": [],
            "category_distribution": [],
            "explanation": "No valid tokens extracted."
        }

    tagged = nltk.pos_tag(clean_tokens)

    items = []
    category_counts: Dict[str, int] = {
        "Noun": 0, "Verb": 0, "Adjective": 0, "Adverb": 0,
        "Pronoun": 0, "Preposition": 0, "Determiner": 0, "Other": 0
    }

    for word, tag in tagged:
        desc, cat = TAG_DESCRIPTIONS.get(tag, (f"Tag ({tag})", "Other"))
        category_counts[cat] = category_counts.get(cat, 0) + 1
        items.append({
            "token": word,
            "tag": tag,
            "description": desc,
            "category": cat
        })

    total = max(1, len(items))
    distribution = [
        {
            "category": cat,
            "count": count,
            "percentage": round((count / total) * 100, 1)
        }
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        if count > 0
    ]

    noun_verb_ratio = round(
        (category_counts.get("Noun", 0) + category_counts.get("Adjective", 0)) /
        max(1, category_counts.get("Verb", 0)),
        2
    )

    explanation = (
        f"Analyzed {len(items)} tokens. Found {category_counts.get('Noun', 0)} nouns and "
        f"{category_counts.get('Adjective', 0)} adjectives relative to {category_counts.get('Verb', 0)} verbs "
        f"(Lexical density ratio: {noun_verb_ratio}). High noun/adjective density typically indicates "
        "substantive conceptual terminology rather than conversational filler phrases."
    )

    return {
        "tokens_with_tags": items,
        "category_distribution": distribution,
        "explanation": explanation
    }

def generate_embeddings_pca(
    student_text: str,
    reference_text: str = "",
    corpus: List[str] = None
) -> Dict[str, Any]:
    """
    Practical Point 4:
    Train lightweight local Word2Vec on academic sentences, project vocabulary into 2D PCA space,
    and compute cosine semantic similarity between student response and reference text.
    """
    # Build text sentences
    raw_texts = [student_text]
    if reference_text:
        raw_texts.append(reference_text)
    if corpus:
        raw_texts.extend(corpus[:25])

    sentences: List[List[str]] = []
    for t in raw_texts:
        if not t:
            continue
        try:
            sents = nltk.sent_tokenize(t)
        except Exception:
            sents = [t]
        for s in sents:
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{2,}\b', s)]
            if len(words) >= 2:
                sentences.append(words)

    if len(sentences) < 2:
        # Fallback sentences for training stability
        sentences.extend([
            ["science", "process", "energy", "system", "law", "motion", "action", "reaction"],
            ["photosynthesis", "plants", "sunlight", "chlorophyll", "glucose", "oxygen"],
            ["respiration", "aerobic", "anaerobic", "energy", "cells", "glucose"],
            ["recursion", "function", "stack", "call", "base", "case", "problem"]
        ])

    try:
        model = Word2Vec(
            sentences=sentences,
            vector_size=24,
            window=3,
            min_count=1,
            workers=1,
            seed=42,
            epochs=20
        )
        vocab = list(model.wv.index_to_key)
    except Exception:
        vocab = []

    if not vocab:
        return {
            "vocab_size": 0,
            "points_2d": [],
            "similar_words": [],
            "semantic_similarity": 0.0,
            "method_used": "Gensim Word2Vec + PCA Projection",
            "explanation": "Insufficient vocabulary tokens to calculate word embeddings."
        }

    # Select top vocabulary words from student text and reference
    student_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', student_text) if w.lower() in model.wv]
    ref_words = [w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', reference_text) if w.lower() in model.wv]

    top_words = list(dict.fromkeys(student_words + ref_words + vocab[:15]))[:20]

    # Project to 2D using PCA
    vectors = np.array([model.wv[w] for w in top_words])
    if len(top_words) >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(vectors)
    else:
        coords = np.zeros((len(top_words), 2))

    points = []
    for i, w in enumerate(top_words):
        points.append({
            "word": w,
            "x": round(float(coords[i][0]), 3),
            "y": round(float(coords[i][1]), 3),
            "freq": student_words.count(w) + ref_words.count(w) + 1
        })

    # Find similar words for the most frequent student term
    similar_words = []
    if student_words:
        focal_word = student_words[0]
        try:
            sims = model.wv.most_similar(focal_word, topn=4)
            similar_words = [{"word": sw, "similarity": round(float(score), 3)} for sw, score in sims]
        except Exception:
            pass

    # Compute sentence-level semantic similarity if reference text exists
    semantic_sim = 0.0
    if student_words and ref_words:
        try:
            s_vec = np.mean([model.wv[w] for w in student_words], axis=0)
            r_vec = np.mean([model.wv[w] for w in ref_words], axis=0)
            dot = np.dot(s_vec, r_vec)
            norm = (np.linalg.norm(s_vec) * np.linalg.norm(r_vec)) + 1e-8
            semantic_sim = round(float(max(0.0, min(1.0, dot / norm))), 3)
        except Exception:
            semantic_sim = 0.5
    elif student_words:
        semantic_sim = 0.5

    explanation = (
        f"Trained a 24-dimensional continuous vector space over {len(vocab)} distinct vocabulary tokens. "
        f"2D PCA coordinates capture relational distances between domain concepts. "
        f"Calculated mean semantic similarity to reference answer: {semantic_sim} (Scale 0 to 1). "
        "Word embeddings map contextual proximity; however, semantic proximity alone does not guarantee "
        "complete factual or logical validity."
    )

    return {
        "vocab_size": len(vocab),
        "points_2d": points,
        "similar_words": similar_words,
        "semantic_similarity": semantic_sim,
        "method_used": "Gensim Word2Vec (Skip-gram/CBOW) + PCA (2D)",
        "explanation": explanation
    }
