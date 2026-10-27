import os
from typing import Dict, Any, List
import pandas as pd

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sample_dataset.csv")

def get_current_dataset(filepath: str = None) -> pd.DataFrame:
    """Load current dataset or default sample dataset."""
    target_path = filepath if filepath and os.path.exists(filepath) else DEFAULT_DATASET_PATH
    if not os.path.exists(target_path):
        return pd.DataFrame(columns=[
            "Student_ID", "Question", "Reference_Answer", "Student_Response", "Subject", "Topic", "Label"
        ])
    return pd.read_csv(target_path)

def get_dataset_summary(filepath: str = None) -> Dict[str, Any]:
    """Generate statistical summary of dataset."""
    df = get_current_dataset(filepath)
    if df.empty:
        return {
            "total_records": 0,
            "columns": [],
            "missing_values": {},
            "label_distribution": {},
            "subject_distribution": {},
            "preview": [],
            "is_synthetic_demonstration": True
        }

    missing = {col: int(df[col].isna().sum()) for col in df.columns}
    label_dist = df["Label"].value_counts().to_dict() if "Label" in df.columns else {}
    subject_dist = df["Subject"].value_counts().to_dict() if "Subject" in df.columns else {}

    preview_rows = df.head(15).fillna("").to_dict(orient="records")

    return {
        "total_records": len(df),
        "columns": list(df.columns),
        "missing_values": missing,
        "label_distribution": {str(k): int(v) for k, v in label_dist.items()},
        "subject_distribution": {str(k): int(v) for k, v in subject_dist.items()},
        "preview": preview_rows,
        "is_synthetic_demonstration": True
    }
