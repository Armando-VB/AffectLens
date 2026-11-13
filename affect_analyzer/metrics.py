"""Scientific affective metrics and dimensional emotion calculations.

Implements James Russell's Circumplex Model of Affect (Valence & Arousal),
Paul Ekman's 7 discrete emotions distribution, affective volatility metrics,
and multi-modal verbal/non-verbal congruence index.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from affect_analyzer.config import (
    AROUSAL_WEIGHTS,
    EKMAN_EMOTIONS,
    QUADRANTS,
    VALENCE_WEIGHTS,
)


@dataclass
class EmotionMetrics:
    """Frame-level computed emotional metrics."""
    timestamp: float                    # Seconds elapsed in session
    emotions: Dict[str, float]          # Raw emotion probabilities [0.0 - 1.0]
    dominant_emotion: str               # Argmax emotion label
    valence: float                      # Continuous valence [-1.0, +1.0]
    arousal: float                      # Continuous arousal [0.0, 1.0]
    quadrant: str                       # Russell Circumplex Quadrant (Q1, Q2, Q3, Q4, Neutral)
    quadrant_name: str                  # Human-readable quadrant label
    face_detected: bool                 # Whether face was found in frame
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)


def normalize_emotions(raw_emotions: Dict[str, float]) -> Dict[str, float]:
    """Ensure all 7 basic emotions exist and sum to 1.0 (probability distribution)."""
    clean = {k: float(raw_emotions.get(k, 0.0)) for k in EKMAN_EMOTIONS}
    total = sum(clean.values())
    if total > 0.0:
        return {k: round(v / total, 4) for k, v in clean.items()}
    # If all zeros (e.g. no detection), return uniform neutral baseline
    baseline = {k: 0.0 for k in EKMAN_EMOTIONS}
    baseline["neutral"] = 1.0
    return baseline


def compute_valence(emotions: Dict[str, float]) -> float:
    """Compute continuous emotional valence in the range [-1.0, 1.0].

    Positive valence reflects pleasant/pleasurable affect (joy, contentment);
    Negative valence reflects unpleasant/aversive affect (fear, anger, sadness, disgust).
    """
    val = sum(emotions.get(emo, 0.0) * VALENCE_WEIGHTS.get(emo, 0.0) for emo in EKMAN_EMOTIONS)
    # Clamp to [-1.0, 1.0]
    return float(np.clip(val, -1.0, 1.0))


def compute_arousal(emotions: Dict[str, float]) -> float:
    """Compute continuous physiological arousal/activation level in the range [0.0, 1.0].

    High arousal reflects alert/energetic states (excitement, terror, rage, shock);
    Low arousal reflects passive/lethargic states (calm, baseline neutrality, sadness).
    """
    arousal = sum(emotions.get(emo, 0.0) * AROUSAL_WEIGHTS.get(emo, 0.0) for emo in EKMAN_EMOTIONS)
    return float(np.clip(arousal, 0.0, 1.0))


def classify_russell_quadrant(
    valence: float, arousal: float, neutrality_threshold: float = 0.15
) -> Tuple[str, str]:
    """Classify emotional state into Russell's Circumplex Affect Quadrants.

    Returns:
        (quadrant_code, quadrant_display_name)
    """
    # Equilibrium zone around center (low arousal and near-zero valence)
    if abs(valence) < neutrality_threshold and arousal < 0.25:
        return "Neutral", QUADRANTS["Neutral"]["name"]

    if valence >= 0.0:
        if arousal >= 0.45:
            return "Q1", QUADRANTS["Q1"]["name"]
        else:
            return "Q4", QUADRANTS["Q4"]["name"]
    else:
        if arousal >= 0.45:
            return "Q2", QUADRANTS["Q2"]["name"]
        else:
            return "Q3", QUADRANTS["Q3"]["name"]


def compute_emotional_volatility(valence_series: List[float]) -> Dict[str, float]:
    """Calculate emotional stability / volatility indices for the session.

    Standard deviation and Mean Absolute Consecutive Difference (MACD).
    Higher values indicate heightened emotional lability or agitation.
    """
    if len(valence_series) < 2:
        return {
            "std_dev": 0.0,
            "mean_consecutive_diff": 0.0,
            "max_swing": 0.0,
            "volatility_score": 0.0,
        }

    arr = np.array(valence_series, dtype=float)
    std_dev = float(np.std(arr))
    diffs = np.abs(np.diff(arr))
    macd = float(np.mean(diffs))
    max_swing = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    # Composite volatility score normalized [0 - 100]
    volatility_score = float(np.clip((std_dev * 50.0) + (macd * 50.0), 0.0, 100.0))

    return {
        "std_dev": round(std_dev, 4),
        "mean_consecutive_diff": round(macd, 4),
        "max_swing": round(max_swing, 4),
        "volatility_score": round(volatility_score, 2),
    }


def detect_emotional_spikes(
    records: List[Dict[str, Any]], delta_threshold: float = 0.30
) -> List[Dict[str, Any]]:
    """Detect significant sudden affective shifts (spikes in Arousal or Negative Valence).

    Useful in academic research to automatically flag moments where interviewees
    experienced sudden surprise, discomfort, stress, or excitement.
    """
    spikes: List[Dict[str, Any]] = []
    if len(records) < 2:
        return spikes

    for i in range(1, len(records)):
        prev = records[i - 1]
        curr = records[i]

        dt = max(curr["timestamp"] - prev["timestamp"], 0.001)
        # Check arousal delta
        arousal_delta = curr["arousal"] - prev["arousal"]
        valence_drop = prev["valence"] - curr["valence"]  # Sudden negativity spike

        rate_arousal = arousal_delta / dt if dt <= 3.0 else arousal_delta
        rate_val_drop = valence_drop / dt if dt <= 3.0 else valence_drop

        if arousal_delta >= delta_threshold or valence_drop >= delta_threshold:
            spike_type = "Arousal Surge" if arousal_delta >= valence_drop else "Negative Affect Drop"
            spikes.append({
                "timestamp": round(curr["timestamp"], 2),
                "type": spike_type,
                "arousal_delta": round(arousal_delta, 3),
                "valence_drop": round(valence_drop, 3),
                "trigger_dominant_emotion": curr.get("dominant_emotion", "unknown"),
                "quadrant": curr.get("quadrant", "Unknown"),
            })

    return spikes


def compute_sentiment_congruence(
    facial_valence: float, text_sentiment_compound: float, threshold: float = 0.40
) -> Dict[str, Any]:
    """Evaluate multimodal congruence between verbal statement and facial expression.

    If an interviewee states something highly positive ("I felt very peaceful and safe",
    compound = +0.8) but their facial valence was negative (-0.5), this indicates
    cognitive dissonance, masking, or psychological suppression.
    """
    discrepancy = abs(facial_valence - text_sentiment_compound)
    is_congruent = discrepancy <= threshold

    interpretation: str
    if is_congruent:
        interpretation = "Congruent: Facial affect aligns with verbal statement."
    elif facial_valence < 0.0 and text_sentiment_compound > 0.2:
        interpretation = (
            "Incongruent (Suppression / Masking): Positive verbal statement masked by negative facial affect."
        )
    elif facial_valence > 0.2 and text_sentiment_compound < -0.2:
        interpretation = (
            "Incongruent (Irony / Nervous Smiling): Negative verbal statement accompanied by smiling or positive affect."
        )
    else:
        interpretation = "Mild Divergence: Facial affect and verbal sentiment show notable variance."

    return {
        "facial_valence": round(facial_valence, 3),
        "text_sentiment": round(text_sentiment_compound, 3),
        "discrepancy_delta": round(discrepancy, 3),
        "is_congruent": is_congruent,
        "interpretation": interpretation,
    }


def summarize_session_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive scientific aggregates for academic reporting."""
    if not records:
        return {
            "total_frames_analyzed": 0,
            "face_detection_rate": 0.0,
            "duration_seconds": 0.0,
            "mean_valence": 0.0,
            "mean_arousal": 0.0,
            "dominant_emotion": "neutral",
            "emotion_proportions": {e: 0.0 for e in EKMAN_EMOTIONS},
            "quadrant_distribution": {q: 0.0 for q in QUADRANTS.keys()},
            "volatility": {"volatility_score": 0.0},
            "spikes_count": 0,
        }

    total_frames = len(records)
    faces_detected = sum(1 for r in records if r.get("face_detected", False))
    face_rate = round(faces_detected / total_frames, 4) if total_frames > 0 else 0.0

    valid_records = [r for r in records if r.get("face_detected", False)]
    if not valid_records:
        valid_records = records

    valences = [r["valence"] for r in valid_records]
    arousals = [r["arousal"] for r in valid_records]

    mean_val = float(np.mean(valences))
    mean_aro = float(np.mean(arousals))

    # Emotion percentages across session
    emotion_sums = {e: 0.0 for e in EKMAN_EMOTIONS}
    for r in valid_records:
        for e in EKMAN_EMOTIONS:
            emotion_sums[e] += r["emotions"].get(e, 0.0)

    n_valid = len(valid_records)
    emotion_proportions = {
        e: round(emotion_sums[e] / n_valid, 4) for e in EKMAN_EMOTIONS
    }

    # Dominant emotion overall
    dominant_emotion = max(emotion_proportions, key=emotion_proportions.get)

    # Russell Quadrant distribution
    quadrant_counts = {q: 0 for q in QUADRANTS.keys()}
    for r in valid_records:
        q = r.get("quadrant", "Neutral")
        if q in quadrant_counts:
            quadrant_counts[q] += 1
        else:
            quadrant_counts["Neutral"] += 1

    quadrant_proportions = {
        q: round(quadrant_counts[q] / n_valid, 4) for q in QUADRANTS.keys()
    }

    volatility = compute_emotional_volatility(valences)
    spikes = detect_emotional_spikes(valid_records)

    start_t = records[0]["timestamp"]
    end_t = records[-1]["timestamp"]
    duration = round(end_t - start_t, 2) if end_t > start_t else round(float(total_frames), 2)

    return {
        "total_frames_analyzed": total_frames,
        "valid_face_frames": faces_detected,
        "face_detection_rate": face_rate,
        "duration_seconds": duration,
        "mean_valence": round(mean_val, 4),
        "mean_arousal": round(mean_aro, 4),
        "dominant_emotion": dominant_emotion,
        "emotion_proportions": emotion_proportions,
        "quadrant_distribution": quadrant_proportions,
        "volatility": volatility,
        "spikes_count": len(spikes),
        "spikes": spikes,
    }
