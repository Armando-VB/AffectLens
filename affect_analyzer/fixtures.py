"""Synthetic Test Fixture & Interview Session Generator.

Generates realistic mock interview video files and emotion telemetry for
automated testing, benchmarks, and research environment verification.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

from affect_analyzer.config import EKMAN_EMOTIONS


def draw_synthetic_face(
    width: int = 400,
    height: int = 400,
    expression: str = "happy",
) -> np.ndarray:
    """Draw a synthetic stylized human face with recognizable facial features."""
    img = np.full((height, width, 3), 245, dtype=np.uint8)

    cx, cy = width // 2, height // 2
    face_radius = int(width * 0.35)

    # Face contour
    cv2.circle(img, (cx, cy), face_radius, (210, 225, 240), -1)
    cv2.circle(img, (cx, cy), face_radius, (100, 100, 100), 2)

    # Eyes
    eye_offset_x = int(face_radius * 0.45)
    eye_offset_y = int(face_radius * 0.25)
    eye_radius = int(face_radius * 0.12)

    left_eye = (cx - eye_offset_x, cy - eye_offset_y)
    right_eye = (cx + eye_offset_x, cy - eye_offset_y)

    cv2.circle(img, left_eye, eye_radius, (255, 255, 255), -1)
    cv2.circle(img, right_eye, eye_radius, (255, 255, 255), -1)
    cv2.circle(img, left_eye, int(eye_radius * 0.5), (60, 40, 20), -1)
    cv2.circle(img, right_eye, int(eye_radius * 0.5), (60, 40, 20), -1)

    # Nose
    nose_pts = np.array([
        [cx, cy - 10],
        [cx - 10, cy + 25],
        [cx + 10, cy + 25],
    ])
    cv2.polylines(img, [nose_pts], True, (150, 150, 150), 2)

    # Mouth based on expression
    mouth_y = cy + int(face_radius * 0.45)
    mouth_w = int(face_radius * 0.5)

    if expression == "happy":
        # Upward curve
        pts = np.array([
            (cx - mouth_w, mouth_y - 10),
            (cx - mouth_w // 2, mouth_y + 20),
            (cx + mouth_w // 2, mouth_y + 20),
            (cx + mouth_w, mouth_y - 10),
        ])
        cv2.polylines(img, [pts], False, (40, 40, 180), 4)
    elif expression == "sad":
        # Downward curve
        pts = np.array([
            (cx - mouth_w, mouth_y + 15),
            (cx - mouth_w // 2, mouth_y - 15),
            (cx + mouth_w // 2, mouth_y - 15),
            (cx + mouth_w, mouth_y + 15),
        ])
        cv2.polylines(img, [pts], False, (180, 40, 40), 4)
    elif expression == "surprise":
        # Open circle mouth
        cv2.circle(img, (cx, mouth_y), int(mouth_w * 0.4), (40, 40, 40), 3)
    else:  # Neutral
        cv2.line(img, (cx - mouth_w, mouth_y), (cx + mouth_w, mouth_y), (60, 60, 60), 3)

    return img


def generate_synthetic_interview_video(
    output_path: str,
    duration_sec: float = 6.0,
    fps: float = 10.0,
) -> str:
    """Create an MP4 test video for pipeline verification.

    Uses real face sample if available (with simulated interview head movements
    and lighting dynamics), or stylized face graphics fallback.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 400, 400
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)
    
    # Check if a real face sample is available in sample_data
    lena_path = Path(__file__).resolve().parent.parent / "sample_data" / "lena.jpg"
    base_face = None
    if lena_path.exists():
        loaded = cv2.imread(str(lena_path))
        if loaded is not None:
            base_face = cv2.resize(loaded, (width, height))

    for f in range(total_frames):
        current_sec = f / fps

        if base_face is not None:
            frame = base_face.copy()
            # Simulate natural interviewee head motion & breathing
            dx = int(np.sin(current_sec * 2.0) * 8.0)
            dy = int(np.cos(current_sec * 1.5) * 5.0)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)

            # Phase-based emotional lighting/contrast shifts
            if 2.0 <= current_sec < 4.0:
                # Dimmer, cooler tone (stress/anxiety phase)
                frame = cv2.convertScaleAbs(frame, alpha=0.88, beta=-15)
            elif current_sec >= 4.0:
                # Brighter, warmer tone (relief phase)
                frame = cv2.convertScaleAbs(frame, alpha=1.05, beta=10)
        else:
            if current_sec < 2.0:
                expr = "neutral"
            elif current_sec < 4.0:
                expr = "sad"
            else:
                expr = "happy"
            frame = draw_synthetic_face(width, height, expr)

        # Add timestamp and phase watermark
        cv2.putText(
            frame,
            f"AffectLens Academic [T: {current_sec:.1f}s]",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        out.write(frame)

    out.release()
    return str(path.resolve())


def generate_mock_interview_timeline(duration_sec: float = 30.0) -> List[Dict[str, Any]]:
    """Generate realistic academic timeline telemetry for fast UI demos and tests."""
    records = []
    np.random.seed(42)

    for s in range(int(duration_sec)):
        t = float(s)
        # Phase 1: Intro (Neutral)
        if t < 8.0:
            emotions = {
                "neutral": 0.70 + np.random.uniform(-0.05, 0.05),
                "happy": 0.15 + np.random.uniform(-0.03, 0.03),
                "surprise": 0.05,
                "sad": 0.04,
                "angry": 0.02,
                "disgust": 0.02,
                "fear": 0.02,
            }
            quad = "Neutral"
            quad_name = "Affective Equilibrium"
        # Phase 2: Sensitive Challenge / Tough Question (Fear/Stress Spike)
        elif t < 18.0:
            emotions = {
                "fear": 0.45 + np.random.uniform(-0.05, 0.05),
                "angry": 0.20 + np.random.uniform(-0.04, 0.04),
                "sad": 0.15,
                "neutral": 0.10,
                "surprise": 0.06,
                "disgust": 0.03,
                "happy": 0.01,
            }
            quad = "Q2"
            quad_name = "High Activation / Negative Affect"
        # Phase 3: Resolution / Positive Rapport (Happy/Calm)
        else:
            emotions = {
                "happy": 0.65 + np.random.uniform(-0.06, 0.06),
                "neutral": 0.20 + np.random.uniform(-0.03, 0.03),
                "surprise": 0.08,
                "sad": 0.02,
                "fear": 0.02,
                "angry": 0.01,
                "disgust": 0.02,
            }
            quad = "Q1"
            quad_name = "High Activation / Positive Affect"

        # Normalize
        tot = sum(emotions.values())
        emotions = {k: round(v / tot, 4) for k, v in emotions.items()}
        dom = max(emotions, key=emotions.get)

        # Valence & Arousal
        from affect_analyzer.metrics import compute_arousal, compute_valence
        val = compute_valence(emotions)
        aro = compute_arousal(emotions)

        records.append({
            "frame_index": s * 30,
            "timestamp": t,
            "face_detected": True,
            "dominant_emotion": dom,
            "dominant_confidence": round(emotions[dom], 3),
            "valence": round(val, 3),
            "arousal": round(aro, 3),
            "quadrant": quad,
            "quadrant_name": quad_name,
            "emotions": emotions,
            "bounding_box": (60, 50, 200, 200),
        })

    return records
