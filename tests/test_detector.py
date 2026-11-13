"""Unit tests for EmotionDetector and frame annotation."""

import numpy as np
import pytest
from affect_analyzer.detector import EmotionDetector
from affect_analyzer.fixtures import draw_synthetic_face


@pytest.fixture
def detector():
    return EmotionDetector()


def test_detector_handles_empty_frame(detector):
    # None frame
    res_none = detector.detect(None)
    assert res_none.face_detected is False
    assert res_none.dominant_emotion == "none"

    # Empty array
    empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
    res_empty = detector.detect(empty_arr)
    assert res_empty.face_detected is False


def test_detector_handles_blank_image(detector):
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detector.detect(blank, is_bgr=True, annotate=True)
    assert res.face_detected is False
    assert res.dominant_emotion == "none"


def test_detector_synthetic_face_annotated(detector):
    face_img = draw_synthetic_face(320, 320, "happy")
    res = detector.detect(face_img, is_bgr=True, annotate=True)

    # Even if FER or Haar detects or falls back, it returns a valid structured result
    assert isinstance(res.emotions, dict)
    assert -1.0 <= res.valence <= 1.0
    assert 0.0 <= res.arousal <= 1.0
    if res.face_detected and res.annotated_frame is not None:
        assert res.annotated_frame.shape == face_img.shape
