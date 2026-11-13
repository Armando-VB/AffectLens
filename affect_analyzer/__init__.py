"""AffectLens Academic - Affective & Emotional Reaction Analysis for Academic Research.

A rigorous Python framework and real-time dashboard for analyzing interviewee
emotional states, facial expressions, Russell's Circumplex affect dynamics (Valence & Arousal),
and state durations in qualitative and behavioral research.
"""

__version__ = "1.0.0"
__author__ = "AffectLens Academic Research Team"

from affect_analyzer.audio_processor import AudioAffectAnalyzer, AudioAffectMetrics
from affect_analyzer.cohort import CohortAggregator, GroupComparisonResult
from affect_analyzer.detector import DetectionResult, EmotionDetector
from affect_analyzer.metrics import (
    EmotionMetrics,
    classify_russell_quadrant,
    compute_arousal,
    compute_emotional_volatility,
    compute_sentiment_congruence,
    compute_valence,
    detect_emotional_spikes,
)
from affect_analyzer.multimodal import MultimodalSessionAnalyzer
from affect_analyzer.report_generator import AcademicReportGenerator
from affect_analyzer.robust_detector import RobustDetectionResult, RobustEmotionDetector
from affect_analyzer.text_processor import TextSentimentAnalyzer
from affect_analyzer.video_processor import VideoAnalysisResult, VideoProcessor

__all__ = [
    "EmotionMetrics",
    "compute_valence",
    "compute_arousal",
    "classify_russell_quadrant",
    "compute_emotional_volatility",
    "detect_emotional_spikes",
    "compute_sentiment_congruence",
    "EmotionDetector",
    "DetectionResult",
    "RobustEmotionDetector",
    "RobustDetectionResult",
    "VideoProcessor",
    "VideoAnalysisResult",
    "TextSentimentAnalyzer",
    "AudioAffectAnalyzer",
    "AudioAffectMetrics",
    "CohortAggregator",
    "GroupComparisonResult",
    "MultimodalSessionAnalyzer",
    "AcademicReportGenerator",
]
