"""Unit tests for multimodal text alignment and report generation."""

import json
import tempfile
from pathlib import Path
import pytest
from affect_analyzer.fixtures import generate_mock_interview_timeline
from affect_analyzer.multimodal import MultimodalSessionAnalyzer
from affect_analyzer.report_generator import AcademicReportGenerator
from affect_analyzer.text_processor import TextSentimentAnalyzer


def test_text_sentiment_analysis():
    analyzer = TextSentimentAnalyzer()

    # Positive statement
    pos_res = analyzer.analyze("I felt really confident, supported, and happy with the result.")
    assert pos_res.compound > 0.4
    assert pos_res.sentiment_label == "Positive"

    # Negative statement
    neg_res = analyzer.analyze("I was terrified and completely overwhelmed by the terrible pressure.")
    assert neg_res.compound < -0.4
    assert neg_res.sentiment_label == "Negative"

    # Hesitation detection
    hes_res = analyzer.analyze("Um, well, I mean... este, I was not sure about that, you know?")
    assert hes_res.hesitation_markers_count >= 3


def test_multimodal_alignment():
    timeline = generate_mock_interview_timeline(duration_sec=20.0)
    questions = [
        {
            "id": "Q1",
            "question": "Tell us about your background",
            "start_sec": 0.0,
            "end_sec": 7.0,
            "verbal_response": "I have been working peacefully in this domain for five years.",
        },
        {
            "id": "Q2",
            "question": "What happened during the crisis?",
            "start_sec": 8.0,
            "end_sec": 17.0,
            "verbal_response": "Everything fell apart, it was stressful and agonizing.",
        },
    ]

    analyzer = MultimodalSessionAnalyzer()
    aligned = analyzer.align_questions(timeline, questions)

    assert len(aligned) == 2
    assert aligned[0].question_id == "Q1"
    assert aligned[1].question_id == "Q2"
    assert aligned[1].mean_facial_arousal > 0.2


def test_academic_report_exports():
    timeline = generate_mock_interview_timeline(duration_sec=15.0)
    metadata = {
        "session_title": "Academic Research Test Interview",
        "subject_id": "P-402",
        "interviewer": "Dr. Smith",
    }
    from affect_analyzer.metrics import summarize_session_metrics
    summary = summarize_session_metrics(timeline)

    # 1. Test CSV export
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    AcademicReportGenerator.export_csv(timeline, csv_path)
    assert Path(csv_path).exists()
    content = Path(csv_path).read_text(encoding="utf-8")
    assert "timestamp_sec" in content
    assert "valence" in content
    Path(csv_path).unlink()

    # 2. Test JSON export
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        json_path = f.name
    AcademicReportGenerator.export_json(metadata, summary, timeline, json_path)
    assert Path(json_path).exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["research_session_metadata"]["subject_id"] == "P-402"
    assert "frame_level_timeline" in loaded
    Path(json_path).unlink()

    # 3. Test Markdown report
    md = AcademicReportGenerator.generate_markdown_report(metadata, summary)
    assert "# Academic Affect Analysis Report" in md
    assert "LaTeX Table Snippet" in md
    assert "\\begin{table}" in md
