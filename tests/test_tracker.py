"""Unit tests for EmotionTracker (TrackerEmociones) in main.py."""

import pytest
from main import EmotionTracker, TrackerEmociones


def test_emotion_tracker_transitions_and_durations():
    tracker = EmotionTracker(umbral_cambio_frames=2)
    tracker.iniciar(t_cero=0.0)

    # State 1: NEUTRAL from 0.0s to 5.0s
    tracker.actualizar("NEUTRAL", 0.9, tiempo_actual=0.0)
    tracker.actualizar("NEUTRAL", 0.85, tiempo_actual=3.0)
    tracker.actualizar("NEUTRAL", 0.9, tiempo_actual=5.0)

    # Transition to HAPPY (requires 2 frames of candidate)
    c1 = tracker.actualizar("HAPPY", 0.95, tiempo_actual=5.5)
    assert c1 is None  # Candidate frame 1

    c2 = tracker.actualizar("HAPPY", 0.92, tiempo_actual=6.0)
    assert c2 is not None  # Confirmed transition
    assert c2["emocion"] in ["NEUTRAL", "Neutral"]
    assert c2["duracion_seg"] == 6.0
    assert c2["inicio_seg"] == 0.0
    assert c2["fin_seg"] == 6.0

    # State 2: HAPPY continues until 10.0s
    tracker.actualizar("HAPPY", 0.90, tiempo_actual=8.0)
    tracker.actualizar("HAPPY", 0.88, tiempo_actual=10.0)

    # Close session at 10.0s
    historial = tracker.cerrar(tiempo_final=10.0)

    assert len(historial) == 2
    assert historial[0]["emocion"] in ["NEUTRAL", "Neutral"]
    assert historial[0]["duracion_seg"] == 6.0

    assert historial[1]["emocion"] in ["HAPPY", "Happy"]
    assert historial[1]["duracion_seg"] == 4.0
    assert historial[1]["inicio_seg"] == 6.0
    assert historial[1]["fin_seg"] == 10.0


def test_tracker_emociones_spanish_alias():
    tracker = TrackerEmociones(umbral_cambio_frames=2)
    tracker.iniciar(t_cero=0.0)

    tracker.actualizar("FELIZ", 0.9, tiempo_actual=0.0)
    tracker.actualizar("FELIZ", 0.9, tiempo_actual=3.0)

    c1 = tracker.actualizar("TRISTE", 0.9, tiempo_actual=4.0)
    assert c1 is None
    c2 = tracker.actualizar("TRISTE", 0.9, tiempo_actual=5.0)
    assert c2 is not None

    historial = tracker.cerrar(tiempo_final=8.0)
    assert len(historial) == 2
