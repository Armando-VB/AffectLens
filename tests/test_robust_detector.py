"""Unit tests for RobustEmotionDetector."""

import cv2
import numpy as np
import pytest
from affect_analyzer.robust_detector import RobustEmotionDetector


@pytest.fixture
def robust_detector():
    return RobustEmotionDetector()


def test_robust_detector_empty_frame(robust_detector):
    res_none = robust_detector.detect(None)
    assert res_none.face_detected is False
    assert res_none.dominant_emotion == "SIN ROSTRO"

    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res_blank = robust_detector.detect(blank)
    assert res_blank.face_detected is False


def test_robust_detector_real_face(robust_detector):
    img = cv2.imread("sample_data/lena.jpg")
    assert img is not None

    res = robust_detector.detect(img)
    assert res.face_detected is True
    assert res.bounding_box is not None
    assert len(res.bounding_box) == 4
    # On Lena's calm resting face, it should predict NEUTRAL, NOT false positive anger or sad!
    assert res.dominant_emotion == "NEUTRAL"
    assert res.dominant_emotion_en == "Neutral"
    assert res.probabilities["neutral"] > 0.50
    assert -1.0 <= res.valence <= 1.0
    assert 0.0 <= res.arousal <= 1.0


def test_robust_detector_calibration_cycle(robust_detector):
    robust_detector.reset_calibration()
    assert robust_detector.is_calibrated is False

    dummy_probs = np.array([0.02, 0.05, 0.01, 0.02, 0.08, 0.75, 0.03, 0.04])
    for _ in range(16):
        calibrated = robust_detector.calibrate_step(dummy_probs)

    assert calibrated is True
    assert robust_detector.is_calibrated is True
    assert robust_detector.baseline_vector is not None
