"""Unit and integration tests for VideoProcessor."""

import tempfile
from pathlib import Path
import pytest
from affect_analyzer.fixtures import generate_synthetic_interview_video
from affect_analyzer.video_processor import VideoProcessor


@pytest.fixture
def sample_video():
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        temp_path = f.name
    generate_synthetic_interview_video(temp_path, duration_sec=3.0, fps=10.0)
    yield temp_path
    Path(temp_path).unlink(missing_ok=True)


def test_video_metadata(sample_video):
    processor = VideoProcessor()
    info = processor.get_video_info(sample_video)
    assert info["duration_sec"] > 2.0
    assert info["fps"] == 10.0
    assert info["width"] == 400
    assert info["height"] == 400


def test_process_video_pipeline(sample_video):
    processor = VideoProcessor()
    progress_updates = []

    def on_progress(ratio, sec):
        progress_updates.append((ratio, sec))

    res = processor.process_video(
        sample_video,
        sampling_fps=2.0,
        progress_callback=on_progress,
        extract_key_frames=True,
    )

    assert res.video_path == sample_video
    assert len(res.timeline) >= 4
    assert len(progress_updates) >= 4
    assert "mean_valence" in res.summary
    assert "mean_arousal" in res.summary
    assert "volatility" in res.summary
