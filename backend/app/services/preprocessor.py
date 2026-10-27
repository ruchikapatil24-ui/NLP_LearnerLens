import re
import string
from typing import Dict, Any, List
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from langdetect import detect, detect_langs

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(nltk_tag: str):
    """Map NLTK POS tag to WordNet POS tag for accurate lemmatization."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted control characters while preserving sentence structure."""
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_language(text: str, user_selected: str = "auto") -> Dict[str, Any]:
    """Detect language with confidence fallback."""
    cleaned = clean_text(text)
    if not cleaned or len(cleaned.split()) < 2:
        return {"language": "en", "name": "English", "confidence": 1.0, "is_fallback": True}

    if user_selected and user_selected != "auto":
        lang_names = {"en": "English", "hi": "Hindi", "mr": "Marathi"}
        return {
            "language": user_selected,
            "name": lang_names.get(user_selected, user_selected.upper()),
            "confidence": 1.0,
            "is_fallback": False
        }

    try:
        langs = detect_langs(cleaned)
        if langs:
            top = langs[0]
            lang_map = {
                "en": "English",
                "hi": "Hindi",
                "mr": "Marathi",
                "es": "Spanish",
                "fr": "French",
                "de": "German"
            }
            return {
                "language": top.lang,
                "name": lang_map.get(top.lang, top.lang.upper()),
                "confidence": round(float(top.prob), 4),
                "is_fallback": False
            }
    except Exception:
        pass
    return {"language": "en", "name": "English", "confidence": 0.85, "is_fallback": True}

def preprocess_response(text: str, user_lang: str = "auto") -> Dict[str, Any]:
    """
    NLP Practical Point 2 & 3:
    Executes language detection, cleaning, word/sentence tokenization,
    calculates vocabulary diversity, and generates Stemming vs Lemmatization comparisons.
    """
    raw_text = text if text else ""
    cleaned = clean_text(raw_text)

    # 1. Language detection
    lang_info = detect_language(cleaned, user_lang)

    # 2. Sentence tokenization
    try:
        sentences = sent_tokenize(cleaned) if cleaned else []
    except Exception:
        sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]

    # 3. Word tokenization
    try:
        all_tokens = word_tokenize(cleaned) if cleaned else []
    except Exception:
        all_tokens = cleaned.split()

    # Define explicit punctuation handling policy:
    # - Lexical words exclude pure punctuation marks and symbols
    # - Hyphenated terms or alphanumeric words are retained
    lexical_words = [t.lower() for t in all_tokens if any(c.isalnum() for c in t)]
    unique_words = sorted(list(set(lexical_words)))

    word_count = len(lexical_words)
    sentence_count = max(1, len(sentences)) if cleaned else 0
    token_count = len(all_tokens)
    unique_count = len(unique_words)

    avg_sentence_len = round(word_count / sentence_count, 2) if sentence_count > 0 else 0.0
    ttr = round(unique_count / max(1, word_count), 4) if word_count > 0 else 0.0

    # 4. POS tagging for lemmatization context
    try:
        nltk_pos_tags = dict(nltk.pos_tag(lexical_words))
    except Exception:
        nltk_pos_tags = {w: "NN" for w in lexical_words}

    # 5. Stemming vs Lemmatization table items
    stem_lemma_comparison: List[Dict[str, str]] = []
    seen = set()

    # Prioritize interesting words with affixes or inflection
    sample_words = []
    for w in lexical_words:
        if w not in seen and len(w) > 2:
            seen.add(w)
            sample_words.append(w)
            if len(sample_words) >= 20:
                break

    for word in sample_words:
        stem = stemmer.stem(word)
        raw_tag = nltk_pos_tags.get(word, "NN")
        wn_pos = get_wordnet_pos(raw_tag)
        try:
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
        except Exception:
            lemma = word

        # Create pedagogical explanation of differences
        if stem != lemma:
            if stem not in wordnet.words() and lemma in wordnet.words():
                explanation = f"Stemming stripped affix heuristically ('{stem}'), whereas lemmatization matched dictionary lemma ('{lemma}')."
            else:
                explanation = f"Affix removal yields stem '{stem}', grammatical reduction yields lemma '{lemma}'."
        elif word != lemma:
            explanation = f"Reduced inflectional ending to canonical form '{lemma}' using POS tag '{raw_tag}'."
        else:
            explanation = "Word is already in canonical root form."

        stem_lemma_comparison.append({
            "original": word,
            "stem": stem,
            "lemma": lemma,
            "pos": raw_tag,
            "explanation": explanation
        })

    counting_policy = {
        "punctuation_policy": "Punctuation marks (.,!?;) are treated as structural sentence tokens but excluded from lexical word count.",
        "contractions_policy": "NLTK Treebank conventions split contractions (e.g. don't -> do + n't) to preserve grammatical verb/negation integrity.",
        "case_handling": "Case normalized to lowercase for vocabulary diversity and unique word counting, preserving original capitalization in display.",
        "ttr_definition": "Type-Token Ratio (TTR) = Unique Lexical Words / Total Lexical Words. Values closer to 1.0 reflect higher vocabulary richness."
    }

    return {
        "original_text": raw_text,
        "cleaned_text": cleaned,
        "detected_language": lang_info["name"],
        "language_confidence": lang_info["confidence"],
        "word_count": word_count,
        "sentence_count": sentence_count,
        "token_count": token_count,
        "unique_word_count": unique_count,
        "avg_sentence_length": avg_sentence_len,
        "vocabulary_diversity": ttr,
        "word_tokens": all_tokens,
        "sentence_tokens": sentences,
        "stem_lemma_comparison": stem_lemma_comparison,
        "counting_policy": counting_policy
    }
