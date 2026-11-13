"""Acoustic Vocal Affect Analysis Engine for Academic Research.

Extracts vocal features from interview audio tracks:
- Root-Mean-Square (RMS) Energy & Vocal Intensity (physiological stress indicator)
- Zero-Crossing Rate (ZCR) & Spectral Centroid proxy (vocal agitation & brightness)
- Speech-to-Pause Ratio & Silence Dynamics (cognitive load / hesitation)
- Acoustic Arousal Estimation
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import scipy.io.wavfile as wavfile

logger = logging.getLogger(__name__)


@dataclass
class AudioAffectMetrics:
    """Acoustic prosody and vocal affect summary for an interview."""
    has_audio: bool
    sample_rate: int
    duration_sec: float
    mean_energy_rms: float
    energy_volatility: float
    mean_zero_crossing_rate: float
    speech_ratio: float                # % of time actively speaking vs silence
    silence_pause_ratio: float         # % of time in pauses/silence
    acoustic_arousal_score: float      # Normalized [0.0, 1.0]
    acoustic_timeline: List[Dict[str, Any]]


class AudioAffectAnalyzer:
    """Analyzes prosodic and acoustic features from video or audio recordings."""

    def __init__(self, frame_length_sec: float = 0.5, silence_threshold_ratio: float = 0.15):
        self.frame_length_sec = frame_length_sec
        self.silence_threshold_ratio = silence_threshold_ratio

    def extract_and_analyze(self, media_path: str) -> AudioAffectMetrics:
        """Extract audio from MP4/WAV/MOV and compute acoustic metrics."""
        path = Path(media_path)
        if not path.exists():
            return self._empty_metrics()

        wav_path = None
        try:
            if path.suffix.lower() == ".wav":
                sr, audio_data = wavfile.read(str(path))
            else:
                # Extract audio from video file
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(str(path))
                if clip.audio is None:
                    return self._empty_metrics()

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    wav_path = tf.name

                clip.audio.write_audiofile(wav_path, fps=16000, nbytes=2, codec="pcm_s16le", logger=None)
                clip.close()
                sr, audio_data = wavfile.read(wav_path)

            return self.analyze_waveform(audio_data, sr)

        except Exception as e:
            logger.warning("Audio processing skipped or unsupported for %s: %s", path.name, e)
            return self._empty_metrics()
        finally:
            if wav_path and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)

    def analyze_waveform(self, audio_data: np.ndarray, sample_rate: int) -> AudioAffectMetrics:
        """Compute acoustic prosody metrics from raw PCM numpy array."""
        # Convert stereo to mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Normalize to float [-1.0, 1.0]
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = float(np.iinfo(audio_data.dtype).max)
            samples = audio_data.astype(np.float32) / max_val
        else:
            samples = audio_data.astype(np.float32)

        total_samples = len(samples)
        duration_sec = total_samples / sample_rate if sample_rate > 0 else 0.0

        if duration_sec <= 0.0:
            return self._empty_metrics()

        hop_length = int(self.frame_length_sec * sample_rate)
        if hop_length <= 0:
            hop_length = 1

        timeline: List[Dict[str, Any]] = []
        energies: List[float] = []
        zcrs: List[float] = []

        for start_idx in range(0, total_samples, hop_length):
            chunk = samples[start_idx : start_idx + hop_length]
            if len(chunk) < 100:
                continue

            t_sec = round(start_idx / sample_rate, 2)
            # Root Mean Square Energy
            rms = float(np.sqrt(np.mean(chunk**2)))
            # Zero Crossing Rate (proxy for vocal frequency content and roughness)
            zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2.0)

            energies.append(rms)
            zcrs.append(zcr)

            timeline.append({
                "timestamp_sec": t_sec,
                "rms_energy": round(rms, 4),
                "zcr": round(zcr, 4),
            })

        if not energies:
            return self._empty_metrics()

        max_e = max(energies) if max(energies) > 0 else 1.0
        norm_energies = [e / max_e for e in energies]
        silence_threshold = self.silence_threshold_ratio

        speech_frames = sum(1 for e in norm_energies if e >= silence_threshold)
        speech_ratio = round(speech_frames / len(energies), 3)
        pause_ratio = round(1.0 - speech_ratio, 3)

        mean_energy = float(np.mean(energies))
        std_energy = float(np.std(energies))
        mean_zcr = float(np.mean(zcrs))

        # Composite acoustic arousal score [0.0 - 1.0]
        acoustic_arousal = float(np.clip((mean_energy * 2.5) + (mean_zcr * 1.5), 0.0, 1.0))

        return AudioAffectMetrics(
            has_audio=True,
            sample_rate=sample_rate,
            duration_sec=round(duration_sec, 2),
            mean_energy_rms=round(mean_energy, 4),
            energy_volatility=round(std_energy, 4),
            mean_zero_crossing_rate=round(mean_zcr, 4),
            speech_ratio=speech_ratio,
            silence_pause_ratio=pause_ratio,
            acoustic_arousal_score=round(acoustic_arousal, 3),
            acoustic_timeline=timeline,
        )

    def _empty_metrics(self) -> AudioAffectMetrics:
        """Fallback when no audio track is found."""
        return AudioAffectMetrics(
            has_audio=False,
            sample_rate=0,
            duration_sec=0.0,
            mean_energy_rms=0.0,
            energy_volatility=0.0,
            mean_zero_crossing_rate=0.0,
            speech_ratio=0.0,
            silence_pause_ratio=0.0,
            acoustic_arousal_score=0.0,
            acoustic_timeline=[],
        )
