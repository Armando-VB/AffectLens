"""Multimodal Alignment Engine for Academic Research.

Synthesizes facial affect timeline data with interview questions, speech turns,
and textual transcripts to measure verbal vs. non-verbal emotional congruence,
detect masking behaviors, and generate question-level research matrices.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np

from affect_analyzer.metrics import compute_sentiment_congruence
from affect_analyzer.text_processor import TextSentimentAnalyzer


@dataclass
class QuestionReactionSummary:
    """Summary of interviewee reaction during a specific interview question/topic."""
    question_id: str
    topic_or_question: str
    timestamp_start: float
    timestamp_end: float
    verbal_response: str
    verbal_sentiment: float
    verbal_label: str
    mean_facial_valence: float
    mean_facial_arousal: float
    dominant_facial_emotion: str
    is_congruent: bool
    congruence_interpretation: str
    arousal_spike_detected: bool


class MultimodalSessionAnalyzer:
    """Combines video affect timeline with question or transcript data."""

    def __init__(self, text_analyzer: Optional[TextSentimentAnalyzer] = None):
        self.text_analyzer = text_analyzer or TextSentimentAnalyzer()

    def align_questions(
        self,
        timeline: List[Dict[str, Any]],
        questions_or_segments: List[Dict[str, Any]],
    ) -> List[QuestionReactionSummary]:
        """Align video frame timeline with question intervals.

        Each segment in questions_or_segments expects:
        - 'id': str (e.g. 'Q1')
        - 'question': str
        - 'start_sec': float
        - 'end_sec': float
        - 'verbal_response': str (optional verbal answer transcript)
        """
        aligned: List[QuestionReactionSummary] = []

        for q in questions_or_segments:
            q_id = str(q.get("id", f"Q{len(aligned)+1}"))
            topic = q.get("question", f"Question {q_id}")
            t_start = float(q.get("start_sec", 0.0))
            t_end = float(q.get("end_sec", t_start + 10.0))
            verbal = q.get("verbal_response", "")

            # Filter timeline within [t_start, t_end]
            sub_records = [
                r for r in timeline
                if t_start <= r["timestamp"] <= t_end and r.get("face_detected", False)
            ]

            if not sub_records:
                # If no detected frames strictly inside, expand search window slightly
                sub_records = [
                    r for r in timeline
                    if (t_start - 2.0) <= r["timestamp"] <= (t_end + 2.0)
                ]

            if sub_records:
                valences = [r["valence"] for r in sub_records]
                arousals = [r["arousal"] for r in sub_records]
                mean_val = float(np.mean(valences))
                mean_aro = float(np.mean(arousals))

                # Count dominant emotions in this segment
                emo_counts: Dict[str, int] = {}
                for r in sub_records:
                    dom = r.get("dominant_emotion", "neutral")
                    emo_counts[dom] = emo_counts.get(dom, 0) + 1
                dom_emotion = max(emo_counts, key=emo_counts.get) if emo_counts else "neutral"

                # Check for arousal spikes
                spike_detected = any(
                    (r["arousal"] - min(arousals)) >= 0.25 for r in sub_records
                )
            else:
                mean_val = 0.0
                mean_aro = 0.05
                dom_emotion = "neutral"
                spike_detected = False

            # Verbal sentiment
            text_res = self.text_analyzer.analyze(verbal)
            congruence = compute_sentiment_congruence(mean_val, text_res.compound)

            aligned.append(QuestionReactionSummary(
                question_id=q_id,
                topic_or_question=topic,
                timestamp_start=t_start,
                timestamp_end=t_end,
                verbal_response=verbal,
                verbal_sentiment=text_res.compound,
                verbal_label=text_res.sentiment_label,
                mean_facial_valence=round(mean_val, 3),
                mean_facial_arousal=round(mean_aro, 3),
                dominant_facial_emotion=dom_emotion,
                is_congruent=congruence["is_congruent"],
                congruence_interpretation=congruence["interpretation"],
                arousal_spike_detected=spike_detected,
            ))

        return aligned
