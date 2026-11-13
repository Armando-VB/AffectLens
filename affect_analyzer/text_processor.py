"""Verbal & Transcript Sentiment Analysis for Academic Interviews.

Uses NLTK VADER sentiment analyzer to evaluate the semantic affect and polarity
of interviewee verbal statements, question-by-question responses, and transcripts.
"""

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class TextSentimentResult:
    """Sentiment results for a verbal statement or interview transcript passage."""
    text: str
    compound: float             # Normalized [-1.0, 1.0]
    positive: float             # [0.0, 1.0]
    neutral: float              # [0.0, 1.0]
    negative: float             # [0.0, 1.0]
    subjectivity_estimate: float # Estimated emotional loading [0.0, 1.0]
    sentiment_label: str        # 'Positive', 'Neutral', 'Negative'
    hesitation_markers_count: int


class TextSentimentAnalyzer:
    """Evaluates verbal sentiment and cognitive hesitation markers in interviews."""

    def __init__(self):
        self._ensure_vader_downloaded()
        self.sia = SentimentIntensityAnalyzer()

        # Common linguistic hesitation / cognitive load markers (English & Spanish)
        self.hesitation_patterns = [
            r"\b(um|uh|erm|err|ah|like|you know|i mean)\b",
            r"\b(este|eh|em|bueno|o sea|es que|digamos)\b",
            r"\.{3,}",  # Ellipsis pauses (...)
        ]

    def _ensure_vader_downloaded(self) -> None:
        """Ensure the VADER lexicon is present."""
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

    def analyze(self, text: str) -> TextSentimentResult:
        """Analyze polarity and affect of an interviewee statement."""
        if not text or not text.strip():
            return TextSentimentResult(
                text="",
                compound=0.0,
                positive=0.0,
                neutral=1.0,
                negative=0.0,
                subjectivity_estimate=0.0,
                sentiment_label="Neutral",
                hesitation_markers_count=0,
            )

        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]
        pos = scores["pos"]
        neu = scores["neu"]
        neg = scores["neg"]

        # Label according to standard academic VADER thresholds
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        # Subjectivity estimate (ratio of polarized affect to neutral content)
        subjectivity = min(pos + neg, 1.0)

        # Count hesitation cues
        hesitations = 0
        for pattern in self.hesitation_patterns:
            matches = re.findall(pattern, text, flags=re.IGNORECASE)
            hesitations += len(matches)

        return TextSentimentResult(
            text=text.strip(),
            compound=round(compound, 3),
            positive=round(pos, 3),
            neutral=round(neu, 3),
            negative=round(neg, 3),
            subjectivity_estimate=round(subjectivity, 3),
            sentiment_label=label,
            hesitation_markers_count=hesitations,
        )

    def analyze_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze a list of interview questions or timestamped transcript turns.

        Expected format per segment:
        {'timestamp_start': float, 'timestamp_end': float, 'speaker': str, 'text': str}
        """
        results: List[Dict[str, Any]] = []
        for seg in segments:
            text = seg.get("text", "")
            sentiment = self.analyze(text)
            record = dict(seg)
            record.update({
                "compound_sentiment": sentiment.compound,
                "sentiment_label": sentiment.sentiment_label,
                "positive": sentiment.positive,
                "neutral": sentiment.neutral,
                "negative": sentiment.negative,
                "hesitation_count": sentiment.hesitation_markers_count,
            })
            results.append(record)
        return results
