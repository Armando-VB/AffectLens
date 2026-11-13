"""Unit tests for AudioAffectAnalyzer and CohortAggregator."""

import numpy as np
import pytest
from affect_analyzer.audio_processor import AudioAffectAnalyzer
from affect_analyzer.cohort import CohortAggregator


def test_audio_analyzer_synthetic_waveform():
    analyzer = AudioAffectAnalyzer(frame_length_sec=0.25)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # 1.0s speech tone (440Hz sine), 1.0s silence
    tone = 0.5 * np.sin(2 * np.pi * 440 * t[: sr])
    silence = np.zeros(sr)
    waveform = np.concatenate([tone, silence]).astype(np.float32)

    res = analyzer.analyze_waveform(waveform, sr)
    assert res.has_audio is True
    assert res.duration_sec == 2.0
    assert 0.3 <= res.speech_ratio <= 0.7
    assert 0.3 <= res.silence_pause_ratio <= 0.7
    assert res.mean_energy_rms > 0.0


def test_cohort_aggregation_and_comparison():
    group_a = [
        {"summary": {"mean_valence": 0.5, "mean_arousal": 0.4, "volatility": {"volatility_score": 10.0}, "face_detection_rate": 0.95, "emotion_proportions": {"happy": 0.6, "neutral": 0.4}}},
        {"summary": {"mean_valence": 0.6, "mean_arousal": 0.5, "volatility": {"volatility_score": 12.0}, "face_detection_rate": 0.98, "emotion_proportions": {"happy": 0.7, "neutral": 0.3}}},
        {"summary": {"mean_valence": 0.55, "mean_arousal": 0.45, "volatility": {"volatility_score": 11.0}, "face_detection_rate": 0.92, "emotion_proportions": {"happy": 0.65, "neutral": 0.35}}},
    ]

    group_b = [
        {"summary": {"mean_valence": -0.4, "mean_arousal": 0.7, "volatility": {"volatility_score": 45.0}, "face_detection_rate": 0.88, "emotion_proportions": {"fear": 0.5, "sad": 0.3, "neutral": 0.2}}},
        {"summary": {"mean_valence": -0.5, "mean_arousal": 0.75, "volatility": {"volatility_score": 50.0}, "face_detection_rate": 0.85, "emotion_proportions": {"fear": 0.6, "sad": 0.2, "neutral": 0.2}}},
        {"summary": {"mean_valence": -0.45, "mean_arousal": 0.8, "volatility": {"volatility_score": 48.0}, "face_detection_rate": 0.90, "emotion_proportions": {"fear": 0.55, "sad": 0.25, "neutral": 0.2}}},
    ]

    agg_a = CohortAggregator.aggregate_sessions(group_a)
    assert agg_a["sample_size"] == 3
    assert agg_a["mean_valence"] > 0.4
    assert agg_a["mean_face_retention"] > 0.9

    comparisons = CohortAggregator.compare_groups(group_a, group_b)
    assert len(comparisons) == 3

    val_comp = next(c for c in comparisons if "Valence" in c.metric_name)
    assert val_comp.statistically_significant is True  # Distinct groups, p < 0.05
    assert val_comp.p_value < 0.01
