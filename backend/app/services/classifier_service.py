import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from .dataset_service import get_current_dataset
from .gap_analyzer import analyze_learning_gaps

SUPPORT_LABEL = "Learner Requiring Support"
ADVANCED_LABEL = "Advanced Understanding"

DISCLAIMER_TEXT = (
    "Preliminary Educational Indicator Only: This prediction is generated from statistical and lexical NLP patterns. "
    "It is NOT a definitive assessment of student intelligence, cognitive capability, or overall potential. "
    "Teachers should use this as a supportive diagnostic tool to guide timely individualized instruction."
)

class ModelRegistry:
    def __init__(self):
        self.vectorizer: TfidfVectorizer = None
        self.svm_model = None
        self.logreg_model = None
        self.nb_model = None
        self.is_trained: bool = False
        self.evaluation_cache: Dict[str, Any] = {}

    def ensure_trained(self):
        if not self.is_trained:
            self.train_and_evaluate_all()

    def train_and_evaluate_all(self, test_size: float = 0.25) -> Dict[str, Any]:
        df = get_current_dataset()
        if df.empty or "Label" not in df.columns or "Student_Response" not in df.columns:
            return {"error": "Dataset is missing required 'Student_Response' or 'Label' columns."}

        # Filter valid labeled data
        valid_df = df[df["Label"].isin([SUPPORT_LABEL, ADVANCED_LABEL])].dropna(subset=["Student_Response", "Label"])
        if len(valid_df) < 6:
            return {"error": f"Insufficient training examples (found {len(valid_df)}, minimum 6 required)."}

        X = valid_df["Student_Response"].values
        y = valid_df["Label"].values

        # Check class distribution
        class_counts = pd.Series(y).value_counts()
        is_imbalanced = bool((class_counts.min() / max(1, class_counts.max())) < 0.65)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except Exception:
            # Fallback if class count is too small for stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        # Vectorize strictly on training split to PREVENT data leakage
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        results = []

        # 1. Multinomial Naive Bayes
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train_vec, y_train)
        y_pred_nb = nb.predict(X_test_vec)
        nb_metrics = self._compute_metrics("Multinomial Naive Bayes", nb, y_test, y_pred_nb, len(y_train), len(y_test), is_imbalanced)
        results.append(nb_metrics)
        self.nb_model = nb

        # 2. Logistic Regression
        lr = LogisticRegression(C=1.0, random_state=42, max_iter=200)
        lr.fit(X_train_vec, y_train)
        y_pred_lr = lr.predict(X_test_vec)
        lr_metrics = self._compute_metrics("Logistic Regression", lr, y_test, y_pred_lr, len(y_train), len(y_test), is_imbalanced)
        results.append(lr_metrics)
        self.logreg_model = lr

        # 3. Linear Support Vector Machine (with calibration for probabilities)
        base_svm = LinearSVC(C=1.0, random_state=42, dual=False, max_iter=2000)
        try:
            # Calibrate SVM to output reliable posterior probabilities
            svm = CalibratedClassifierCV(estimator=base_svm, cv=2)
            svm.fit(X_train_vec, y_train)
        except Exception:
            svm = base_svm
            svm.fit(X_train_vec, y_train)

        y_pred_svm = svm.predict(X_test_vec)
        svm_metrics = self._compute_metrics("Linear SVM (Calibrated)", svm, y_test, y_pred_svm, len(y_train), len(y_test), is_imbalanced)
        results.append(svm_metrics)
        self.svm_model = svm

        self.is_trained = True

        eval_summary = {
            "models": results,
            "dataset_size": len(valid_df),
            "test_size": test_size,
            "evaluation_notes": (
                f"Evaluated on {len(y_test)} held-out test responses. "
                "Vectorizers were fitted strictly on training data to prevent data leakage. "
                + ("Note: Class distribution is mildly imbalanced; F1-score provides a more reliable metric than raw accuracy." if is_imbalanced else "")
            )
        }
        self.evaluation_cache = eval_summary
        return eval_summary

    def _compute_metrics(self, name: str, model, y_true, y_pred, train_count: int, test_count: int, is_imbalanced: bool) -> Dict[str, Any]:
        acc = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[SUPPORT_LABEL, ADVANCED_LABEL])

        # Per-class report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        note = "Balanced evaluation split."
        if is_imbalanced:
            note = "Classes are imbalanced; prioritize macro F1 over raw accuracy."

        return {
            "model_name": name,
            "accuracy": round(acc, 3),
            "precision": round(float(p), 3),
            "recall": round(float(r), 3),
            "f1_score": round(float(f1), 3),
            "confusion_matrix": cm.tolist(),
            "class_report": report,
            "train_count": int(train_count),
            "test_count": int(test_count),
            "is_imbalanced": bool(is_imbalanced),
            "note": str(note)
        }

    def predict_response(
        self,
        student_text: str,
        reference_text: str = "",
        model_name: str = "Linear SVM (Calibrated)"
    ) -> Dict[str, Any]:
        self.ensure_trained()

        clean_text = student_text.strip()
        words = clean_text.split()

        # Rule-based fallback if ML model is unavailable or text is too short/anomalous
        if not self.is_trained or not self.vectorizer or len(words) < 4:
            return self._predict_rule_based(clean_text, reference_text)

        vec = self.vectorizer.transform([clean_text])

        # Pick chosen model
        chosen = self.svm_model
        if "Naive Bayes" in model_name and self.nb_model:
            chosen = self.nb_model
        elif "Logistic" in model_name and self.logreg_model:
            chosen = self.logreg_model

        if chosen is None:
            chosen = self.svm_model or self.logreg_model or self.nb_model

        try:
            pred_label = str(chosen.predict(vec)[0])
            if hasattr(chosen, "predict_proba"):
                probs = chosen.predict_proba(vec)[0]
                classes = list(chosen.classes_)
                scores = {cls: round(float(prob), 3) for cls, prob in zip(classes, probs)}
                confidence = float(scores.get(pred_label, 0.75))
            else:
                decision = float(chosen.decision_function(vec)[0])
                prob_est = round(1.0 / (1.0 + np.exp(-decision)), 3)
                scores = {ADVANCED_LABEL: prob_est, SUPPORT_LABEL: round(1.0 - prob_est, 3)}
                confidence = round(max(prob_est, 1.0 - prob_est), 3)
        except Exception:
            return self._predict_rule_based(clean_text, reference_text)

        gap_analysis = analyze_learning_gaps(clean_text, reference_text)

        return {
            "predicted_category": pred_label,
            "confidence": round(confidence, 3),
            "model_name": model_name,
            "classification_type": "Supervised ML (TF-IDF + Linear Classifier)",
            "decision_scores": scores,
            "rule_triggers": gap_analysis["rule_triggers"],
            "learning_gaps": gap_analysis["learning_gaps"],
            "recommendations": gap_analysis["recommendations"],
            "disclaimer": DISCLAIMER_TEXT
        }

    def _predict_rule_based(self, student_text: str, reference_text: str) -> Dict[str, Any]:
        words = student_text.split()
        word_count = len(words)
        gap_analysis = analyze_learning_gaps(student_text, reference_text)

        # Rule heuristics
        support_triggers = []
        if word_count < 14:
            support_triggers.append("Brief answer (< 14 words)")
        if any(c in student_text.lower() for c in ["stuff", "things", "like if", "bad countries"]):
            support_triggers.append("Informal colloquial terminology")

        if len(support_triggers) > 0:
            pred = SUPPORT_LABEL
            conf = 0.80
        else:
            pred = ADVANCED_LABEL
            conf = 0.75

        scores = {
            SUPPORT_LABEL: 0.80 if pred == SUPPORT_LABEL else 0.20,
            ADVANCED_LABEL: 0.75 if pred == ADVANCED_LABEL else 0.25
        }

        return {
            "predicted_category": pred,
            "confidence": conf,
            "model_name": "Rule-Based Pedagogical Indicator",
            "classification_type": "Rule-based Demonstration",
            "decision_scores": scores,
            "rule_triggers": gap_analysis["rule_triggers"] + support_triggers,
            "learning_gaps": gap_analysis["learning_gaps"],
            "recommendations": gap_analysis["recommendations"],
            "disclaimer": DISCLAIMER_TEXT
        }

model_registry = ModelRegistry()
