"""Video Processing & Temporal Affect Tracking Engine.

Ingests academic interview recordings, samples frames according to the scientific
sampling rate, calculates temporal affect dynamics, and produces structured
timeline datasets.
"""

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np

from affect_analyzer.config import AnalysisConfig
from affect_analyzer.detector import EmotionDetector
from affect_analyzer.metrics import summarize_session_metrics

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    """Complete analysis dataset for an interview video session."""
    video_path: str
    metadata: Dict[str, Any]
    timeline: List[Dict[str, Any]]
    summary: Dict[str, Any]
    key_frame_samples: List[Dict[str, Any]]


class VideoProcessor:
    """Processes video sessions with configurable sampling and progress reporting."""

    def __init__(
        self,
        detector: Optional[EmotionDetector] = None,
        config: Optional[AnalysisConfig] = None,
    ):
        self.config = config or AnalysisConfig()
        self.detector = detector or EmotionDetector(self.config)

    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata without loading entire file into memory."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0

        cap.release()
        return {
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 2),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
        }

    def process_video(
        self,
        video_path: str,
        sampling_fps: Optional[float] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        max_duration_sec: Optional[float] = None,
        extract_key_frames: bool = True,
    ) -> VideoAnalysisResult:
        """Run temporal affect analysis across the interview video.

        Args:
            video_path: Absolute or relative path to MP4/AVI/MOV video.
            sampling_fps: Frames to analyze per second (e.g., 1.0 or 2.0).
            progress_callback: Optional function(ratio [0.0-1.0], current_sec).
            max_duration_sec: Optional cap on seconds to analyze (e.g. for preview).
            extract_key_frames: Save sample annotated RGB frames for visual reports.

        Returns:
            VideoAnalysisResult containing timeline and summary statistics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        meta = self.get_video_info(video_path)
        native_fps = meta["fps"]
        total_duration = meta["duration_sec"]
        if max_duration_sec:
            total_duration = min(total_duration, max_duration_sec)

        rate = sampling_fps or self.config.sampling_fps
        frame_step = max(int(round(native_fps / rate)), 1)

        timeline: List[Dict[str, Any]] = []
        key_frames: List[Dict[str, Any]] = []

        frame_idx = 0
        analyzed_count = 0
        total_steps = int(round(total_duration * rate)) if total_duration > 0 else 100

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_sec = frame_idx / native_fps
            if max_duration_sec and current_sec > max_duration_sec:
                break

            if frame_idx % frame_step == 0:
                analyzed_count += 1
                should_annotate = extract_key_frames and (
                    len(key_frames) < 12 or analyzed_count % max(int(total_steps / 8), 1) == 0
                )

                det_res = self.detector.detect(
                    frame, is_bgr=True, annotate=should_annotate
                )

                record = {
                    "frame_index": frame_idx,
                    "timestamp": round(current_sec, 2),
                    "face_detected": det_res.face_detected,
                    "dominant_emotion": det_res.dominant_emotion,
                    "dominant_confidence": det_res.dominant_confidence,
                    "valence": det_res.valence,
                    "arousal": det_res.arousal,
                    "quadrant": det_res.quadrant,
                    "quadrant_name": det_res.quadrant_name,
                    "emotions": det_res.emotions,
                    "bounding_box": det_res.bounding_box,
                }
                timeline.append(record)

                if should_annotate and det_res.annotated_frame is not None:
                    rgb_annotated = cv2.cvtColor(det_res.annotated_frame, cv2.COLOR_BGR2RGB)
                    key_frames.append({
                        "timestamp": round(current_sec, 2),
                        "dominant_emotion": det_res.dominant_emotion,
                        "valence": det_res.valence,
                        "arousal": det_res.arousal,
                        "frame_rgb": rgb_annotated,
                    })

                if progress_callback:
                    ratio = min(current_sec / max(total_duration, 0.001), 1.0)
                    progress_callback(ratio, current_sec)

            frame_idx += 1

        cap.release()

        # Apply moving average smoothing to valence/arousal to stabilize academic graphs
        smoothed_timeline = self._apply_temporal_smoothing(timeline)

        # Generate scientific summary metrics
        summary = summarize_session_metrics(smoothed_timeline)

        return VideoAnalysisResult(
            video_path=video_path,
            metadata=meta,
            timeline=smoothed_timeline,
            summary=summary,
            key_frame_samples=key_frames,
        )

    def _apply_temporal_smoothing(
        self, timeline: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply a rolling window filter to reduce high-frequency camera/lighting noise."""
        if len(timeline) < self.config.smoothing_window:
            return timeline

        w = self.config.smoothing_window
        valences = [r["valence"] for r in timeline]
        arousals = [r["arousal"] for r in timeline]

        # Convolutional moving average with edge padding
        kernel = np.ones(w) / w
        smooth_val = np.convolve(valences, kernel, mode="same")
        smooth_aro = np.convolve(arousals, kernel, mode="same")

        for idx, rec in enumerate(timeline):
            rec["raw_valence"] = rec["valence"]
            rec["raw_arousal"] = rec["arousal"]
            rec["valence"] = round(float(smooth_val[idx]), 3)
            rec["arousal"] = round(float(smooth_aro[idx]), 3)

        return timeline
