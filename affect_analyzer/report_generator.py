"""Academic Research Report & Dataset Exporter.

Generates publication-ready artifacts:
- Standard CSV datasets for statistical analysis (R, SPSS, Python pandas)
- Comprehensive JSON schema for reproducible research
- Academic Executive Summary in Markdown with LaTeX table snippets.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """Encodes numpy scalar types and arrays into standard JSON primitives."""
    def default(self, obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


class AcademicReportGenerator:
    """Exports structured datasets and research synthesis reports."""

    @staticmethod
    def generate_csv_dataframe(timeline: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert frame-level timeline into an analysis-ready pandas DataFrame."""
        rows = []
        for r in timeline:
            row = {
                "timestamp_sec": float(r.get("timestamp", 0.0)),
                "face_detected": bool(r.get("face_detected", False)),
                "dominant_emotion": str(r.get("dominant_emotion", "none")),
                "dominant_confidence": float(r.get("dominant_confidence", 0.0)),
                "valence": float(r.get("valence", 0.0)),
                "arousal": float(r.get("arousal", 0.0)),
                "russell_quadrant": str(r.get("quadrant", "Neutral")),
                "quadrant_name": str(r.get("quadrant_name", "Neutral")),
            }
            emotions = r.get("emotions", {})
            for emo, prob in emotions.items():
                row[f"prob_{emo}"] = float(prob)
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def export_csv(timeline: List[Dict[str, Any]], output_path: str) -> str:
        """Write timeline dataset to CSV file."""
        df = AcademicReportGenerator.generate_csv_dataframe(timeline)
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return str(p.resolve())

    @staticmethod
    def export_json(
        metadata: Dict[str, Any],
        summary: Dict[str, Any],
        timeline: List[Dict[str, Any]],
        output_path: str,
        question_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Export complete scientific JSON document."""
        data = {
            "research_session_metadata": metadata,
            "scientific_summary": summary,
            "question_level_affect": question_summaries or [],
            "frame_level_timeline": timeline,
        }
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        return str(p.resolve())

    @staticmethod
    def generate_markdown_report(
        metadata: Dict[str, Any],
        summary: Dict[str, Any],
        question_summaries: Optional[List[Any]] = None,
    ) -> str:
        """Compile a publication-grade scientific Markdown synthesis report."""
        session_title = metadata.get("session_title", "Interview Affective Analysis")
        subject_id = metadata.get("subject_id", "Participant-01")
        duration = summary.get("duration_seconds", 0.0)
        face_rate = summary.get("face_detection_rate", 0.0) * 100.0
        mean_val = summary.get("mean_valence", 0.0)
        mean_aro = summary.get("mean_arousal", 0.0)
        dominant_emo = summary.get("dominant_emotion", "neutral")
        volatility = summary.get("volatility", {})
        spikes = summary.get("spikes", [])
        proportions = summary.get("emotion_proportions", {})
        quadrants = summary.get("quadrant_distribution", {})

        md = []
        md.append(f"# Academic Affect Analysis Report: {session_title}")
        md.append("")
        md.append("## 1. Executive Research Summary")
        md.append(f"- **Participant / Subject ID:** `{subject_id}`")
        md.append(f"- **Session Duration:** `{duration:.1f}` seconds ({duration/60.0:.2f} min)")
        md.append(f"- **Facial Tracking Reliability:** `{face_rate:.1f}%` of sampled frames")
        md.append(f"- **Primary Affective Baseline:** **{dominant_emo.upper()}**")
        md.append(f"- **Mean Continuous Valence:** `{mean_val:+.3f}` (scale [-1.0, +1.0])")
        md.append(f"- **Mean Continuous Arousal:** `{mean_aro:.3f}` (scale [0.0, 1.0])")
        md.append(f"- **Affective Volatility Score (EVI):** `{volatility.get('volatility_score', 0.0):.1f} / 100` (StdDev: `{volatility.get('std_dev', 0.0):.3f}`)")
        md.append(f"- **Acute Emotional Spikes Detected:** `{len(spikes)}` events")
        md.append("")

        md.append("## 2. Paul Ekman 7-Discrete Emotions Distribution")
        md.append("| Affective State | Proportion (%) | Scientific Classification |")
        md.append("|:---|:---:|:---|")
        for emo, prop in sorted(proportions.items(), key=lambda x: x[1], reverse=True):
            pct = prop * 100.0
            classification = "Positive Affect" if emo == "happy" else ("Neutral Baseline" if emo == "neutral" else "Negative / Aversive Affect")
            md.append(f"| **{emo.capitalize()}** | {pct:.2f}% | {classification} |")
        md.append("")

        md.append("## 3. Russell's Circumplex Affect Model Residency")
        md.append("| Circumplex Quadrant | Proportion (%) | Affective Profile |")
        md.append("|:---|:---:|:---|")
        for q, prop in quadrants.items():
            pct = prop * 100.0
            md.append(f"| **{q}** | {pct:.2f}% | High/Low Activation & Valence profile |")
        md.append("")

        if spikes:
            md.append("## 4. Significant Affective Spikes & Micro-Reactions")
            md.append("Acute shifts in interviewee stress, surprise, or emotional agitation:")
            md.append("| Timestamp (s) | Event Classification | Trigger Emotion | Arousal $\\Delta$ | Valence Drop |")
            md.append("|:---:|:---|:---|:---:|:---:|")
            for sp in spikes:
                md.append(
                    f"| {sp['timestamp']:.1f}s | {sp['type']} | {sp['trigger_dominant_emotion']} | +{sp['arousal_delta']:.2f} | {sp['valence_drop']:.2f} |"
                )
            md.append("")

        if question_summaries:
            md.append("## 5. Question-by-Question Reaction & Verbal Congruence Matrix")
            md.append("| Question / Topic | Dominant Face Emotion | Face Valence | Verbal Sentiment | Congruence Assessment |")
            md.append("|:---|:---:|:---:|:---:|:---|")
            for q in question_summaries:
                # Can be QuestionReactionSummary object or dict
                q_topic = getattr(q, "topic_or_question", None) or q.get("topic_or_question", "Question")
                q_emo = getattr(q, "dominant_facial_emotion", None) or q.get("dominant_facial_emotion", "neutral")
                q_val = getattr(q, "mean_facial_valence", None) or q.get("mean_facial_valence", 0.0)
                q_verb = getattr(q, "verbal_sentiment", None) or q.get("verbal_sentiment", 0.0)
                q_interp = getattr(q, "congruence_interpretation", None) or q.get("congruence_interpretation", "N/A")

                md.append(f"| {q_topic} | {q_emo} | {q_val:+.2f} | {q_verb:+.2f} | {q_interp} |")
            md.append("")

        md.append("## 6. Academic Paper LaTeX Table Snippet")
        md.append("```latex")
        md.append("\\begin{table}[htbp]")
        md.append("\\centering")
        md.append(f"\\caption{{Affective Dynamics Summary for Participant {subject_id}}}")
        md.append("\\begin{tabular}{lccc}")
        md.append("\\hline")
        md.append("\\textbf{Affect Metric} & \\textbf{Observed Mean} & \\textbf{Std. Dev.} & \\textbf{Baseline} \\\\")
        md.append("\\hline")
        md.append(f"Valence (V) & {mean_val:+.3f} & {volatility.get('std_dev', 0.0):.3f} & [-1.0, +1.0] \\\\")
        md.append(f"Arousal (A) & {mean_aro:.3f} & - & [0.0, 1.0] \\\\")
        md.append(f"Dominant Emotion & {dominant_emo.capitalize()} ({proportions.get(dominant_emo, 0.0)*100:.1f}\\%) & - & Ekman 7 \\\\")
        md.append(f"Emotional Volatility & {volatility.get('volatility_score', 0.0):.1f} & - & [0, 100] \\\\")
        md.append(f"Face Retention & {face_rate:.1f}\\% & - & [0\\%, 100\\%] \\\\")
        md.append("\\hline")
        md.append("\\end{tabular}")
        md.append("\\end{table}")
        md.append("```")

        return "\n".join(md)
