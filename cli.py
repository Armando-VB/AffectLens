"""AffectLens CLI - Academic Affect & Reaction Analyzer.

Command-line interface for batch processing interview recordings,
generating reproducible scientific reports, and running diagnostics.
"""

import argparse
from pathlib import Path
import sys

from affect_analyzer.config import AnalysisConfig
from affect_analyzer.fixtures import (
    generate_mock_interview_timeline,
    generate_synthetic_interview_video,
)
from affect_analyzer.metrics import summarize_session_metrics
from affect_analyzer.multimodal import MultimodalSessionAnalyzer
from affect_analyzer.report_generator import AcademicReportGenerator
from affect_analyzer.video_processor import VideoProcessor


def analyze_video_cmd(args):
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir or video_path.parent / f"{video_path.stem}_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig(sampling_fps=args.fps)
    processor = VideoProcessor(config=config)

    print(f"[AffectLens] Analyzing video: {video_path}")
    print(f"[AffectLens] Sampling rate: {args.fps} FPS")

    def progress(ratio, current_sec):
        pct = int(ratio * 100)
        bar = "=" * (pct // 5) + ">"
        sys.stdout.write(f"\rProgress: [{bar:<20}] {pct}% ({current_sec:.1f}s)")
        sys.stdout.flush()

    res = processor.process_video(
        str(video_path),
        sampling_fps=args.fps,
        progress_callback=progress,
        max_duration_sec=args.max_duration,
    )
    print("\n[AffectLens] Video processing complete. Compiling reports...")

    meta = {
        "session_title": args.title or video_path.stem,
        "subject_id": args.subject or "Participant-01",
        "video_path": str(video_path.resolve()),
        "native_fps": res.metadata["fps"],
        "resolution": res.metadata["resolution"],
    }

    # Export CSV
    csv_file = output_dir / f"{video_path.stem}_timeline.csv"
    AcademicReportGenerator.export_csv(res.timeline, str(csv_file))

    # Export JSON
    json_file = output_dir / f"{video_path.stem}_dataset.json"
    AcademicReportGenerator.export_json(meta, res.summary, res.timeline, str(json_file))

    # Export Markdown
    md_file = output_dir / f"{video_path.stem}_report.md"
    md_content = AcademicReportGenerator.generate_markdown_report(meta, res.summary)
    md_file.write_text(md_content, encoding="utf-8")

    print(f"\n=======================================================")
    print(f" AFFECTLENS ACADEMIC ANALYSIS SUMMARY")
    print(f"=======================================================")
    print(f" Subject ID:            {meta['subject_id']}")
    print(f" Total Duration:        {res.summary['duration_seconds']:.1f} s")
    print(f" Face Retention Rate:   {res.summary['face_detection_rate']*100:.1f}%")
    print(f" Dominant Emotion:      {res.summary['dominant_emotion'].upper()}")
    print(f" Mean Valence (V):      {res.summary['mean_valence']:+.3f} [-1.0, +1.0]")
    print(f" Mean Arousal (A):      {res.summary['mean_arousal']:.3f} [0.0, 1.0]")
    print(f" Volatility Score:      {res.summary['volatility']['volatility_score']:.1f} / 100")
    print(f" Affective Spikes:      {res.summary['spikes_count']}")
    print(f" Artifacts generated in: {output_dir.resolve()}")
    print(f" - CSV Dataset:         {csv_file.name}")
    print(f" - JSON Schema:         {json_file.name}")
    print(f" - Research Report:     {md_file.name}")
    print(f"=======================================================")


def generate_fixture_cmd(args):
    out = Path(args.output or "sample_data/sample_interview.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[AffectLens] Generating synthetic test interview video ({args.duration}s)...")
    res_path = generate_synthetic_interview_video(str(out), duration_sec=args.duration)
    print(f"[AffectLens] Synthetic test video ready at: {res_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AffectLens Academic - Facial & Multimodal Interview Reaction Analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # analyze-video
    p_video = subparsers.add_parser("analyze-video", help="Analyze an interview video file.")
    p_video.add_argument("--video", "-v", required=True, help="Path to video file.")
    p_video.add_argument("--fps", type=float, default=1.0, help="Sampling FPS (default: 1.0).")
    p_video.add_argument("--output-dir", "-o", help="Directory to save datasets and reports.")
    p_video.add_argument("--subject", default="Participant-01", help="Participant ID.")
    p_video.add_argument("--title", default="Academic Interview", help="Study or session title.")
    p_video.add_argument("--max-duration", type=float, help="Max duration in seconds to process.")
    p_video.set_defaults(func=analyze_video_cmd)

    # generate-fixture
    p_fix = subparsers.add_parser("generate-fixture", help="Generate synthetic test video fixture.")
    p_fix.add_argument("--output", "-o", default="sample_data/sample_interview.mp4", help="Output path.")
    p_fix.add_argument("--duration", type=float, default=6.0, help="Duration in seconds.")
    p_fix.set_defaults(func=generate_fixture_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
