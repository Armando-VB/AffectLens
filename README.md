# 🔬 AffectLens AI: Real-Time Interview Facial Affect & Sentiment Analyzer

> **Open-Source Python Platform for Academic, Psychological, and Qualitative Research**  
> Enables continuous, objective tracking of an interviewee's emotional state, cognitive reflection, and affective valence during interviews using real-time computer vision, deep affective feature extraction (AffectNet ONNX), resting-face baseline calibration, and automated research reporting.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python: 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests: 24 Passed](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)]()
[![Inference: ~19ms CPU](https://img.shields.io/badge/inference-~19ms%20CPU-orange.svg)]()

---

## 🎯 The Research Problem & Scientific Objective

In qualitative research, behavioral psychology, investigative interviewing, and UX/HCI studies, assessing an interviewee's genuine emotional reaction is typically hindered by:
1. **Subjective Observer Bias:** Different researchers interpret subtle facial cues differently.
2. **Resting-Face Bias ("Resting Bitch Face" / RBF):** Natural facial bone structure and resting neutral expressions are frequently misclassified by standard AI models as angry, sullen, or contemptuous.
3. **Cognitive Load vs. Negative Affect:** Subject Matter Experts (SMEs) furrow their brows when concentrating or reflecting on difficult questions, causing conventional models to mistakenly flag hostility or anger.

**AffectLens solves this** with an academic-grade architecture:
- **Baseline Neutral Calibration:** Automatically calibrates against the participant's unique neutral resting face during the first 2-3 seconds of each session.
- **Deep Cognitive Load Disambiguation (`THINKING`):** Distinguishes between authentic anger/frustration and deep reflective cognitive processing.
- **Russell's Circumplex Model (1980):** Quantifies continuous emotional trajectory across **Valence** ($V \in [-1.0, +1.0]$, hedonic tone) and **Arousal** ($A \in [0.0, 1.0]$, physiological activation).
- **Automated Research Deliverables:** Generates structured dataset CSVs, executive qualitative reports, and publication-ready timeline charts.

---

## 🚀 Quick Start (Terminal MVP)

### 1. Run live with your webcam
```bash
python3 main.py
```

1. Review the initial guidance and press **`ENTER`** to start recording (PLAY).
2. The real-time interactive interface displays:
   - **Video Feed:** Live camera with bounding box, calibrated affect label, and confidence score.
   - **Real-Time Analytics HUD:** Dynamic probability bars across 8 emotional states (`Neutral`, `Happy`, `Sad`, `Surprise`, `Thinking`, `Anger`, `Disgust`, `Fear`) and a real-time continuous **Valence trajectory waveform**.
3. **Session Controls:**
   - **`[SPACE]`**: Bookmark a specific question or notable interview moment.
   - **`[p]`**: Pause or resume recording.
   - **`[q]`**: Complete session, stop tracking, and export all analytical artifacts.

---

## 📊 Automated Research Deliverables

At the conclusion of each session, AffectLens automatically generates:

| Artifact | Format | Description |
| :--- | :--- | :--- |
| **`emotion_records.csv`** | CSV Table | Granular chronological sequence with exact `start_sec`, `end_sec`, `duration_sec`, detected `emotion`, and mean confidence score. Ready for R, SPSS, or Pandas. |
| **`session_summary.txt`** | Text Report | Academic executive summary detailing total session duration, percentage distribution per emotion, affective volatility, and timestamped question bookmarks. |
| **`session_timeline_chart.png`** | PNG (300 DPI) | High-resolution publication-grade chart depicting emotional state transitions and continuous valence trajectory over time. |

---

## ⚙️ CLI & Headless Execution

AffectLens can run seamlessly on pre-recorded interview videos or in headless automated pipelines:

```bash
# Analyze a pre-recorded interview video file
python3 main.py --video sample_data/sample_interview.mp4

# Run in headless mode (no GUI window, terminal output only)
python3 main.py --video sample_data/sample_interview.mp4 --no-gui --auto-start

# Specify custom export paths
python3 main.py --video interview_p01.mp4 --output-csv p01_records.csv --output-summary p01_summary.txt --output-chart p01_chart.png
```

---

## 🧠 Technical Architecture

```text
                  +-----------------------------------+
                  |         Video / Webcam Input      |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------------------------+
                  |      UltraFace ONNX / Haar        |
                  |     Real-Time Face Detection      |
                  +-----------------+-----------------+
                                    | (Cropped Face)
                                    v
                  +-----------------------------------+
                  |   AffectNet EfficientNet-B0 ONNX   |
                  |   Deep Feature Affect Classifier  |
                  +-----------------+-----------------+
                                    |
            +-----------------------+-----------------------+
            |                                               |
            v                                               v
+-----------------------+                       +-----------------------+
|  Resting-Face Baseline|                       | Russell Circumplex &  |
|  Dynamic Calibration  |                       | Continuous Valence    |
+-----------+-----------+                       +-----------+-----------+
            |                                               |
            +-----------------------+-----------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |       Emotion State Tracker       |
                  |  State Transitions & Durations    |
                  +-----------------+-----------------+
                                    |
         +--------------------------+--------------------------+
         |                          |                          |
         v                          v                          v
+------------------+      +-------------------+      +-------------------+
|  Structured CSV  |      |  Academic Summary |      |  Timeline Chart   |
| emotion_records  |      |  session_summary  |      | session_timeline  |
+------------------+      +-------------------+      +-------------------+
```

- **Core Model:** AffectNet pre-trained EfficientNet-B0 running on ONNX Runtime (~19ms inference latency on CPU).
- **Fallback Engine:** Haar Cascade + FER MiniXception engine ensures zero-dependency reliability in any environment.
- **Metrics Engine:** Calculates continuous Emotional Volatility Index (EVI), affective spikes ($|\Delta V| > 0.40$), and Russell dimensional coordinates.

---

## 🧪 Automated Testing & Verification

The test suite covers metrics, detectors, calibration cycles, tracking state transitions, and end-to-end video pipelines with 100% pass rate:

```bash
python3 -m pytest -v
```

```text
tests/test_advanced.py::test_audio_analyzer_synthetic_waveform PASSED
tests/test_advanced.py::test_cohort_aggregation_and_comparison PASSED
tests/test_cli.py::test_cli_generate_fixture_and_analyze PASSED
tests/test_detector.py::test_detector_handles_empty_frame PASSED
tests/test_detector.py::test_detector_handles_blank_image PASSED
tests/test_detector.py::test_detector_synthetic_face_annotated PASSED
tests/test_metrics.py::test_normalize_emotions PASSED
tests/test_metrics.py::test_compute_valence PASSED
tests/test_metrics.py::test_compute_arousal PASSED
tests/test_metrics.py::test_classify_russell_quadrant PASSED
tests/test_metrics.py::test_emotional_volatility PASSED
tests/test_metrics.py::test_detect_emotional_spikes PASSED
tests/test_metrics.py::test_sentiment_congruence PASSED
tests/test_metrics.py::test_summarize_session_metrics PASSED
tests/test_multimodal.py::test_text_sentiment_analysis PASSED
tests/test_multimodal.py::test_multimodal_alignment PASSED
tests/test_multimodal.py::test_academic_report_exports PASSED
tests/test_robust_detector.py::test_robust_detector_empty_frame PASSED
tests/test_robust_detector.py::test_robust_detector_real_face PASSED
tests/test_robust_detector.py::test_robust_detector_calibration_cycle PASSED
tests/test_tracker.py::test_emotion_tracker_transitions_and_durations PASSED
tests/test_tracker.py::test_tracker_emociones_spanish_alias PASSED
tests/test_video.py::test_video_metadata PASSED
tests/test_video.py::test_process_video_pipeline PASSED

======================== 24 passed in 11.81s ========================
```

---

## 📁 Repository Structure

```text
Sentiment_AI/
├── affect_analyzer/
│   ├── __init__.py           # Package exports
│   ├── robust_detector.py    # AffectNet ONNX + Resting-Face Baseline Calibration
│   ├── detector.py           # Haar + FER fallback emotion detector
│   ├── metrics.py            # Russell circumplex (Valence, Arousal, EVI, Spikes)
│   ├── video_processor.py    # Video file processing & temporal smoothing
│   ├── text_processor.py     # Speech sentiment & disfluency analysis
│   ├── multimodal.py         # Multimodal interview alignment
│   ├── audio_processor.py    # Acoustic waveform affect analyzer
│   ├── cohort.py             # Cross-session group comparison
│   ├── report_generator.py   # Academic CSV, JSON, Markdown, LaTeX exporter
│   └── fixtures.py           # Synthetic video & benchmark fixture generator
├── sample_data/
│   ├── enet_b0_8_best_afew.onnx  # AffectNet EfficientNet-B0 ONNX weights
│   ├── ultraface_slim_320.onnx   # UltraFace face detector weights
│   └── sample_interview.mp4      # Demonstration interview video
├── tests/                    # 24 automated unit and integration tests
├── main.py                   # Terminal MVP: Live webcam & video analyzer
├── cli.py                    # Scriptable CLI automation
├── app.py                    # Streamlit web UI for researchers
├── requirements.txt          # Python dependencies
├── LICENSE                   # Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)
└── README.md                 # Technical documentation & project guide
```

---

## 📜 License & Usage Terms

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

- **Permitted:** Academic research, educational use, university coursework, thesis projects, and personal non-commercial study.
- **Prohibited:** Commercial exploitation, selling, licensing, monetizing, or incorporating this software into commercial fee-charging products or services without prior written permission.

See [LICENSE](LICENSE) for full legal text or visit [creativecommons.org/licenses/by-nc/4.0](https://creativecommons.org/licenses/by-nc/4.0/).
