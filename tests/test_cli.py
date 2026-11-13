"""Integration tests for AffectLens CLI commands."""

import subprocess
import sys
import tempfile
from pathlib import Path


def test_cli_generate_fixture_and_analyze():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        video_file = tmp_path / "test_session.mp4"
        out_dir = tmp_path / "analysis_results"

        # 1. Generate synthetic fixture
        gen_cmd = [
            sys.executable,
            "cli.py",
            "generate-fixture",
            "--output",
            str(video_file),
            "--duration",
            "3.0",
        ]
        res_gen = subprocess.run(gen_cmd, capture_output=True, text=True)
        assert res_gen.returncode == 0
        assert video_file.exists()

        # 2. Analyze video
        ana_cmd = [
            sys.executable,
            "cli.py",
            "analyze-video",
            "--video",
            str(video_file),
            "--fps",
            "2.0",
            "--output-dir",
            str(out_dir),
            "--subject",
            "Participant-Test-Auto",
        ]
        res_ana = subprocess.run(ana_cmd, capture_output=True, text=True)
        assert res_ana.returncode == 0
        assert "AFFECTLENS ACADEMIC ANALYSIS SUMMARY" in res_ana.stdout

        # Verify output files
        csv_file = out_dir / f"{video_file.stem}_timeline.csv"
        json_file = out_dir / f"{video_file.stem}_dataset.json"
        report_file = out_dir / f"{video_file.stem}_report.md"

        assert csv_file.exists()
        assert json_file.exists()
        assert report_file.exists()
