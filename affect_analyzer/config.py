"""Academic Affect Configuration and Constants.

Definitions adhere to Paul Ekman's Basic Emotion Model and James Russell's
Circumplex Model of Affect (Valence-Arousal plane).
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Ekman's 7 primary basic emotions
EKMAN_EMOTIONS: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Color palette for academic charts and visualizations
EMOTION_COLORS: Dict[str, str] = {
    "happy": "#2ecc71",     # Green
    "sad": "#3498db",       # Blue
    "angry": "#e74c3c",     # Red
    "fear": "#9b59b6",      # Purple
    "disgust": "#e67e22",   # Orange
    "surprise": "#f1c40f",  # Yellow
    "neutral": "#95a5a6",   # Gray
}

# Russell's Circumplex Affect Quadrants
QUADRANTS: Dict[str, Dict[str, str]] = {
    "Q1": {
        "name": "High Activation / Positive Affect",
        "description": "Enthusiastic, Joyful, Excited, Motivated",
        "color": "#27ae60",
    },
    "Q2": {
        "name": "High Activation / Negative Affect",
        "description": "Stressed, Anxious, Frustrated, Agitated",
        "color": "#c0392b",
    },
    "Q3": {
        "name": "Low Activation / Negative Affect",
        "description": "Melancholic, Bored, Dejected, Fatigued",
        "color": "#2980b9",
    },
    "Q4": {
        "name": "Low Activation / Positive Affect",
        "description": "Calm, Serene, Relaxed, Content",
        "color": "#16a085",
    },
    "Neutral": {
        "name": "Affective Equilibrium",
        "description": "Baseline, Objective, Unaroused",
        "color": "#7f8c8d",
    },
}

# Valence calculation weights (normalized [-1.0, 1.0])
# Positive: happy (+1.0), surprise (+0.1)
# Negative: sad (-0.8), angry (-0.9), fear (-0.85), disgust (-0.7)
VALENCE_WEIGHTS: Dict[str, float] = {
    "happy": 1.0,
    "surprise": 0.15,
    "neutral": 0.0,
    "disgust": -0.65,
    "sad": -0.75,
    "fear": -0.85,
    "angry": -0.95,
}

# Arousal (physiological activation) calculation weights [0.0, 1.0]
AROUSAL_WEIGHTS: Dict[str, float] = {
    "surprise": 0.90,
    "fear": 0.85,
    "angry": 0.80,
    "happy": 0.70,
    "sad": 0.35,
    "disgust": 0.40,
    "neutral": 0.05,
}


@dataclass
class AnalysisConfig:
    """Configurable parameters for academic session analysis."""
    sampling_fps: float = 1.0            # Analyze 1 frame per second (configurable)
    min_confidence: float = 0.20         # Minimum emotion probability threshold
    spike_threshold: float = 0.35        # Sudden delta jump in arousal/negative valence to flag as spike
    smoothing_window: int = 3            # Moving average window size (frames)
    congruence_threshold: float = 0.40   # Delta between text sentiment and facial valence for incongruence
    face_min_size: int = 50              # Minimum bounding box dimension in pixels
