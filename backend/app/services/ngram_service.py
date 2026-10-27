import re
import math
from collections import Counter
from typing import Dict, Any, List

def tokenize_clean_words(text: str) -> List[str]:
    """Tokenize words into clean lowercase alphabetical tokens."""
    return [w.lower() for w in re.findall(r'\b[a-zA-Z]{1,}\b', text)]

def calculate_ngrams(
    text: str,
    n_order: int = 2,
    reference_corpus: List[str] = None
) -> Dict[str, Any]:
    """
    Practical Point 7:
    Generate unigrams, bigrams, and trigrams.
    Compute observed frequencies, MLE conditional sequence probabilities,
    Laplace add-1 smoothing, and sequence log probability/perplexity.
    """
    words = tokenize_clean_words(text)
    if not words:
        return {
            "unigrams": [],
            "bigrams": [],
            "trigrams": [],
            "sequence_log_prob": 0.0,
            "perplexity": 0.0,
            "laplace_applied": False,
            "explanation": "No words available for N-Gram modeling."
        }

    # Build reference context words if provided
    ref_words = []
    if reference_corpus:
        for doc in reference_corpus:
            ref_words.extend(tokenize_clean_words(doc))

    # Background model combines words
    model_words = words + ref_words
    total_tokens = len(model_words)
    unigram_counts = Counter(model_words)
    vocab_size = len(unigram_counts)

    # 1. Unigrams
    unigrams = [
        {
            "ngram": w,
            "frequency": count,
            "probability": round(count / total_tokens, 4)
        }
        for w, count in unigram_counts.most_common(12)
    ]

    # 2. Bigrams
    bigram_tuples = list(zip(model_words[:-1], model_words[1:]))
    bigram_counts = Counter(bigram_tuples)
    bigrams = []
    for (w1, w2), count in bigram_counts.most_common(12):
        cond_prob = count / max(1, unigram_counts[w1])
        bigrams.append({
            "ngram": f"{w1} -> {w2}",
            "frequency": count,
            "probability": round(cond_prob, 4)
        })

    # 3. Trigrams
    trigram_tuples = list(zip(model_words[:-2], model_words[1:-1], model_words[2:]))
    trigram_counts = Counter(trigram_tuples)
    trigrams = []
    for (w1, w2, w3), count in trigram_counts.most_common(10):
        prev_bigram_count = bigram_counts.get((w1, w2), 1)
        cond_prob = count / max(1, prev_bigram_count)
        trigrams.append({
            "ngram": f"{w1} {w2} -> {w3}",
            "frequency": count,
            "probability": round(cond_prob, 4)
        })

    # Sequence probability calculation with Laplace Smoothing
    # P(w1...wn) = prod P(wi | wi-1)
    student_bigrams = list(zip(words[:-1], words[1:]))
    log_prob_sum = 0.0
    v = max(10, vocab_size)

    for w1, w2 in student_bigrams:
        count_pair = bigram_counts.get((w1, w2), 0)
        count_w1 = unigram_counts.get(w1, 0)
        # Laplace smoothing: (count + 1) / (count_w1 + V)
        prob = (count_pair + 1.0) / (count_w1 + v)
        log_prob_sum += math.log2(prob)

    num_eval_pairs = max(1, len(student_bigrams))
    avg_log_prob = log_prob_sum / num_eval_pairs
    perplexity = round(math.pow(2, -avg_log_prob), 2)

    explanation = (
        f"Generated {len(unigrams)} unigrams, {len(bigrams)} bigrams, and {len(trigrams)} trigrams. "
        f"Evaluated {len(student_bigrams)} phrase transitions. "
        f"Laplace add-1 smoothing was applied across vocabulary size V={v} to prevent zero probability traps. "
        f"Calculated Sequence Perplexity: {perplexity}. Lower perplexity indicates predictable, syntactically fluent "
        "collocations; higher perplexity signals novel or fragmented phrasing."
    )

    return {
        "unigrams": unigrams,
        "bigrams": bigrams,
        "trigrams": trigrams,
        "sequence_log_prob": round(log_prob_sum, 2),
        "perplexity": perplexity,
        "laplace_applied": True,
        "explanation": explanation
    }
