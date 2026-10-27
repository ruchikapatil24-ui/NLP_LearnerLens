from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AnalyzeRequest(BaseModel):
    student_id: Optional[str] = "STU_ANON"
    subject: Optional[str] = "General Science"
    question: str
    reference_answer: Optional[str] = ""
    student_response: str
    language: Optional[str] = "auto"

class PreprocessRequest(BaseModel):
    text: str
    language: Optional[str] = "auto"

class POSRequest(BaseModel):
    text: str

class NERRequest(BaseModel):
    text: str

class TFIDFRequest(BaseModel):
    student_text: str
    reference_text: Optional[str] = ""
    corpus: Optional[List[str]] = None

class NgramRequest(BaseModel):
    text: str
    n_order: Optional[int] = 2
    reference_corpus: Optional[List[str]] = None

class LSIRequest(BaseModel):
    student_text: str
    reference_text: Optional[str] = ""
    n_components: Optional[int] = 2

class StemLemmaItem(BaseModel):
    original: str
    stem: str
    lemma: str
    pos: str
    explanation: str

class PreprocessResponse(BaseModel):
    original_text: str
    cleaned_text: str
    detected_language: str
    language_confidence: float
    word_count: int
    sentence_count: int
    token_count: int
    unique_word_count: int
    avg_sentence_length: float
    vocabulary_diversity: float
    word_tokens: List[str]
    sentence_tokens: List[str]
    stem_lemma_comparison: List[StemLemmaItem]
    counting_policy: Dict[str, str]

class POSTagItem(BaseModel):
    token: str
    tag: str
    description: str
    category: str

class POSResponse(BaseModel):
    tokens_with_tags: List[POSTagItem]
    category_distribution: List[Dict[str, Any]]
    explanation: str

class WordVectorPoint(BaseModel):
    word: str
    x: float
    y: float
    freq: int

class EmbeddingsResponse(BaseModel):
    vocab_size: int
    points_2d: List[WordVectorPoint]
    similar_words: List[Dict[str, Any]]
    semantic_similarity: float
    method_used: str
    explanation: str

class EntityItem(BaseModel):
    text: str
    label: str
    description: str
    start: int
    end: int

class NERResponse(BaseModel):
    original_text: str
    entities: List[EntityItem]
    category_counts: List[Dict[str, Any]]
    model_source: str
    explanation: str

class TFIDFTerm(BaseModel):
    term: str
    score: float
    in_reference: bool

class TFIDFResponse(BaseModel):
    top_terms: List[TFIDFTerm]
    reference_similarity: float
    vocabulary_size: int
    matrix_heatmap: Optional[List[Dict[str, Any]]] = None
    formula_explanation: str
    interpretation: str

class NgramItem(BaseModel):
    ngram: str
    frequency: int
    probability: float

class NgramResponse(BaseModel):
    unigrams: List[NgramItem]
    bigrams: List[NgramItem]
    trigrams: List[NgramItem]
    sequence_log_prob: float
    perplexity: float
    laplace_applied: bool
    explanation: str

class LSIResponse(BaseModel):
    explained_variance_ratio: List[float]
    student_coordinates: List[float]
    reference_coordinates: Optional[List[float]] = None
    similarity_score: float
    latent_topics: List[Dict[str, Any]]
    explanation: str
    short_answer_warning: Optional[str] = None

class ClassificationResponse(BaseModel):
    predicted_category: str  # "Learner Requiring Support" or "Advanced Understanding"
    confidence: float
    model_name: str
    classification_type: str  # "Supervised ML" or "Rule-based Demonstration"
    decision_scores: Dict[str, float]
    rule_triggers: List[str]
    learning_gaps: List[str]
    recommendations: List[str]
    disclaimer: str

class FullAnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    student_id: str
    subject: str
    question: str
    reference_answer: str
    student_response: str
    preprocess: PreprocessResponse
    pos: POSResponse
    embeddings: EmbeddingsResponse
    ner: NERResponse
    tfidf: TFIDFResponse
    ngram: NgramResponse
    lsi: LSIResponse
    classification: ClassificationResponse
    pedagogical_insights: Dict[str, Any]

class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_report: Dict[str, Any]
    train_count: int
    test_count: int
    is_imbalanced: bool
    note: str

class ModelComparisonResponse(BaseModel):
    models: List[ModelMetrics]
    dataset_size: int
    test_size: float
    evaluation_notes: str

class DatasetSummaryResponse(BaseModel):
    total_records: int
    columns: List[str]
    missing_values: Dict[str, int]
    label_distribution: Dict[str, int]
    subject_distribution: Dict[str, int]
    preview: List[Dict[str, Any]]
    is_synthetic_demonstration: bool
