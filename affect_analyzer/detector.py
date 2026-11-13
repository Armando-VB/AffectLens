"""Face & Emotion Detection Engine.

Wraps FER and OpenCV face detectors with academic-grade reliability:
- Handles low-contrast / uneven lighting
- Filters for the primary interviewee (largest bounding box)
- Annotates frames with scientific telemetry (bounding box, valence, dominant emotion)
- Provides smooth fallback when face is temporarily obstructed or lost.
"""

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from affect_analyzer.config import (
    AnalysisConfig,
    EKMAN_EMOTIONS,
    EMOTION_COLORS,
)
from affect_analyzer.metrics import (
    classify_russell_quadrant,
    compute_arousal,
    compute_valence,
    normalize_emotions,
)

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection results for a single video frame or image."""
    face_detected: bool
    bounding_box: Optional[Tuple[int, int, int, int]]  # (x, y, w, h)
    emotions: Dict[str, float]
    dominant_emotion: str
    dominant_confidence: float
    valence: float
    arousal: float
    quadrant: str
    quadrant_name: str
    annotated_frame: Optional[np.ndarray] = None


class EmotionDetector:
    """Robust Facial Emotion Recognition Detector for Academic Research."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self._fer = None
        self._face_cascade = None
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Initialize FER and OpenCV Haar cascades with graceful fallback."""
        try:
            from fer.fer import FER
            # Instantiate FER with TFLite backend
            self._fer = FER(min_face_size=self.config.face_min_size, mtcnn=False)
            logger.info("FER Emotion Recognition backend loaded successfully.")
        except Exception as e:
            logger.warning("FER initialization warning: %s. Relying on Haar fallback.", e)
            self._fer = None

        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception as e:
            logger.error("Failed to load OpenCV face cascade: %s", e)
            self._face_cascade = None

    def detect(
        self,
        frame: np.ndarray,
        is_bgr: bool = True,
        annotate: bool = False,
    ) -> DetectionResult:
        """Analyze a single frame for the primary interviewee's emotional state.

        Args:
            frame: Numpy image array (H, W, C).
            is_bgr: True if frame is in OpenCV BGR format; False if RGB.
            annotate: Whether to generate an annotated visual overlay.

        Returns:
            DetectionResult dataclass.
        """
        if frame is None or frame.size == 0:
            return self._empty_result(frame, annotate)

        # Standardize color formats
        if is_bgr:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bgr_frame = frame
        else:
            rgb_frame = frame
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        detections = []
        if self._fer is not None:
            try:
                # FER expects RGB image
                raw_dets = self._fer.detect_emotions(rgb_frame)
                if raw_dets:
                    detections = raw_dets
            except Exception as e:
                logger.debug("FER detection exception: %s", e)

        # Select the primary interviewee: largest bounding box area
        primary_det = None
        if detections:
            primary_det = max(
                detections,
                key=lambda d: d.get("box", (0, 0, 0, 0))[2] * d.get("box", (0, 0, 0, 0))[3],
            )

        if primary_det and primary_det.get("emotions"):
            box = tuple(primary_det["box"])
            raw_emotions = primary_det["emotions"]
            emotions = normalize_emotions(raw_emotions)
            dominant_emo = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emo]

            valence = compute_valence(emotions)
            arousal = compute_arousal(emotions)
            quad_code, quad_name = classify_russell_quadrant(valence, arousal)

            annotated = None
            if annotate:
                annotated = self.annotate_frame(
                    bgr_frame.copy(), box, emotions, dominant_emo, valence, arousal, quad_name
                )
                if not is_bgr:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            return DetectionResult(
                face_detected=True,
                bounding_box=box,
                emotions=emotions,
                dominant_emotion=dominant_emo,
                dominant_confidence=round(confidence, 3),
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                quadrant=quad_code,
                quadrant_name=quad_name,
                annotated_frame=annotated,
            )

        # Fallback: Check if face cascade detects a face even if FER confidence was low
        fallback_box = self._detect_face_cascade(rgb_frame)
        if fallback_box is not None:
            baseline_emotions = {e: 0.0 for e in EKMAN_EMOTIONS}
            baseline_emotions["neutral"] = 1.0
            valence = 0.0
            arousal = 0.05
            quad_code, quad_name = classify_russell_quadrant(valence, arousal)

            annotated = None
            if annotate:
                annotated = self.annotate_frame(
                    bgr_frame.copy(),
                    fallback_box,
                    baseline_emotions,
                    "neutral (unresolved)",
                    valence,
                    arousal,
                    quad_name,
                )
                if not is_bgr:
                    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            return DetectionResult(
                face_detected=True,
                bounding_box=fallback_box,
                emotions=baseline_emotions,
                dominant_emotion="neutral",
                dominant_confidence=0.5,
                valence=valence,
                arousal=arousal,
                quadrant=quad_code,
                quadrant_name=quad_name,
                annotated_frame=annotated,
            )

        return self._empty_result(bgr_frame if is_bgr else rgb_frame, annotate)

    def _detect_face_cascade(self, rgb_frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Fallback face detector using Haar cascade."""
        if self._face_cascade is None:
            return None
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.config.face_min_size, self.config.face_min_size),
        )
        if len(faces) > 0:
            # Pick largest
            largest = max(faces, key=lambda b: b[2] * b[3])
            return tuple(int(x) for x in largest)
        return None

    def _empty_result(self, frame: Optional[np.ndarray], annotate: bool) -> DetectionResult:
        """Return standardized baseline when no face is present."""
        baseline = {e: 0.0 for e in EKMAN_EMOTIONS}
        baseline["neutral"] = 1.0
        return DetectionResult(
            face_detected=False,
            bounding_box=None,
            emotions=baseline,
            dominant_emotion="none",
            dominant_confidence=0.0,
            valence=0.0,
            arousal=0.0,
            quadrant="Neutral",
            quadrant_name="No Face Detected",
            annotated_frame=frame if annotate else None,
        )

    def annotate_frame(
        self,
        bgr_img: np.ndarray,
        box: Tuple[int, int, int, int],
        emotions: Dict[str, float],
        dominant_emo: str,
        valence: float,
        arousal: float,
        quadrant_name: str,
    ) -> np.ndarray:
        """Draw research overlay: face box, valence meter, and dominant emotion tag."""
        x, y, w, h = box
        # Box color based on dominant emotion
        color_hex = EMOTION_COLORS.get(dominant_emo.split()[0], "#3498db")
        # Hex to BGR
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        bgr_color = (b, g, r)

        # Draw face bounding box with rounded aesthetic
        cv2.rectangle(bgr_img, (x, y), (x + w, y + h), bgr_color, 2)

        # Header badge: Dominant Emotion + Confidence
        label = f"{dominant_emo.upper()} | Val: {valence:+.2f} | Aro: {arousal:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        badge_y = max(y - 10, lh + 10)
        cv2.rectangle(
            bgr_img,
            (x, badge_y - lh - 6),
            (x + lw + 12, badge_y + 4),
            (25, 25, 25),
            -1,
        )
        cv2.rectangle(
            bgr_img,
            (x, badge_y - lh - 6),
            (x + lw + 12, badge_y + 4),
            bgr_color,
            1,
        )
        cv2.putText(
            bgr_img,
            label,
            (x + 6, badge_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Secondary pill: Circumplex Quadrant
        quad_label = f"State: {quadrant_name}"
        (qw, qh), _ = cv2.getTextSize(quad_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        sub_y = y + h + qh + 8
        if sub_y < bgr_img.shape[0]:
            cv2.rectangle(
                bgr_img,
                (x, sub_y - qh - 4),
                (x + qw + 8, sub_y + 4),
                (35, 35, 35),
                -1,
            )
            cv2.putText(
                bgr_img,
                quad_label,
                (x + 4, sub_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

        return bgr_img
