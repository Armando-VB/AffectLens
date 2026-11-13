"""Unit tests for scientific emotion metrics and affect calculations."""

import pytest
from affect_analyzer.metrics import (
    classify_russell_quadrant,
    compute_arousal,
    compute_emotional_volatility,
    compute_sentiment_congruence,
    compute_valence,
    detect_emotional_spikes,
    normalize_emotions,
    summarize_session_metrics,
)


def test_normalize_emotions():
    raw = {"happy": 2.0, "sad": 2.0, "neutral": 6.0}
    norm = normalize_emotions(raw)
    assert abs(sum(norm.values()) - 1.0) < 1e-4
    assert norm["happy"] == 0.20
    assert norm["sad"] == 0.20
    assert norm["neutral"] == 0.60
    assert norm["angry"] == 0.0


def test_compute_valence():
    happy_state = {"happy": 1.0, "neutral": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0, "surprise": 0.0}
    sad_state = {"sad": 1.0, "happy": 0.0, "neutral": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0, "surprise": 0.0}
    neutral_state = {"neutral": 1.0, "happy": 0.0, "sad": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0, "surprise": 0.0}

    assert compute_valence(happy_state) > 0.8
    assert compute_valence(sad_state) < -0.6
    assert abs(compute_valence(neutral_state)) < 0.01


def test_compute_arousal():
    surprise_state = {"surprise": 1.0, "neutral": 0.0, "sad": 0.0, "happy": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0}
    neutral_state = {"neutral": 1.0, "surprise": 0.0, "sad": 0.0, "happy": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0}

    assert compute_arousal(surprise_state) > 0.8
    assert compute_arousal(neutral_state) < 0.15


def test_classify_russell_quadrant():
    # Q1: High valence, high arousal
    q, name = classify_russell_quadrant(0.6, 0.7)
    assert q == "Q1"

    # Q2: Low valence, high arousal
    q, name = classify_russell_quadrant(-0.5, 0.6)
    assert q == "Q2"

    # Q3: Low valence, low arousal
    q, name = classify_russell_quadrant(-0.5, 0.3)
    assert q == "Q3"

    # Q4: High valence, low arousal
    q, name = classify_russell_quadrant(0.5, 0.3)
    assert q == "Q4"

    # Neutral center
    q, name = classify_russell_quadrant(0.02, 0.1)
    assert q == "Neutral"


def test_emotional_volatility():
    # Stable series
    stable_series = [0.1, 0.12, 0.11, 0.09, 0.10]
    res_stable = compute_emotional_volatility(stable_series)
    assert res_stable["volatility_score"] < 10.0

    # Highly volatile series
    volatile_series = [0.8, -0.7, 0.9, -0.8, 0.7]
    res_vol = compute_emotional_volatility(volatile_series)
    assert res_vol["volatility_score"] > 50.0
    assert res_vol["std_dev"] > 0.6


def test_detect_emotional_spikes():
    timeline = [
        {"timestamp": 1.0, "valence": 0.1, "arousal": 0.1, "dominant_emotion": "neutral", "quadrant": "Neutral"},
        {"timestamp": 2.0, "valence": -0.6, "arousal": 0.8, "dominant_emotion": "fear", "quadrant": "Q2"},
        {"timestamp": 3.0, "valence": 0.2, "arousal": 0.2, "dominant_emotion": "neutral", "quadrant": "Neutral"},
    ]
    spikes = detect_emotional_spikes(timeline, delta_threshold=0.35)
    assert len(spikes) >= 1
    assert spikes[0]["timestamp"] == 2.0
    assert spikes[0]["trigger_dominant_emotion"] == "fear"


def test_sentiment_congruence():
    # Congruent: positive face, positive words
    res1 = compute_sentiment_congruence(facial_valence=0.7, text_sentiment_compound=0.8)
    assert res1["is_congruent"] is True

    # Incongruent: smiling when saying something very negative
    res2 = compute_sentiment_congruence(facial_valence=0.6, text_sentiment_compound=-0.7)
    assert res2["is_congruent"] is False
    assert "Incongruent" in res2["interpretation"]

    # Incongruent: distressed face when saying "I am fine"
    res3 = compute_sentiment_congruence(facial_valence=-0.6, text_sentiment_compound=0.7)
    assert res3["is_congruent"] is False
    assert "Masking" in res3["interpretation"]


def test_summarize_session_metrics():
    # Empty
    empty_summary = summarize_session_metrics([])
    assert empty_summary["total_frames_analyzed"] == 0

    # With records
    records = [
        {
            "timestamp": float(i),
            "face_detected": True,
            "valence": 0.4 if i % 2 == 0 else 0.2,
            "arousal": 0.3,
            "dominant_emotion": "happy" if i % 2 == 0 else "neutral",
            "quadrant": "Q4",
            "emotions": {"happy": 0.6, "neutral": 0.4, "sad": 0.0, "angry": 0.0, "fear": 0.0, "disgust": 0.0, "surprise": 0.0},
        }
        for i in range(10)
    ]
    summary = summarize_session_metrics(records)
    assert summary["total_frames_analyzed"] == 10
    assert summary["face_detection_rate"] == 1.0
    assert summary["duration_seconds"] == 9.0
    assert summary["mean_valence"] > 0.0
