"""AffectLens Academic - Interview Affect & Reaction Analysis Dashboard.

Streamlit web application designed for academic behavioral and qualitative researchers.
Enables video affect analysis, multimodal transcript alignment, live session monitoring,
and publication-ready dataset exports.
"""

import io
import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from affect_analyzer.audio_processor import AudioAffectAnalyzer
from affect_analyzer.cohort import CohortAggregator
from affect_analyzer.config import (
    AnalysisConfig,
    EKMAN_EMOTIONS,
    EMOTION_COLORS,
    QUADRANTS,
)
from affect_analyzer.detector import EmotionDetector
from affect_analyzer.fixtures import (
    generate_mock_interview_timeline,
    generate_synthetic_interview_video,
)
from affect_analyzer.metrics import (
    classify_russell_quadrant,
    compute_arousal,
    compute_valence,
    summarize_session_metrics,
)
from affect_analyzer.multimodal import MultimodalSessionAnalyzer
from affect_analyzer.report_generator import AcademicReportGenerator
from affect_analyzer.text_processor import TextSentimentAnalyzer
from affect_analyzer.video_processor import VideoProcessor

# Page configuration
st.set_page_config(
    page_title="AffectLens Academic | Interview Reaction Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Academic Grade Interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .badge-q1 { background-color: #dcfce7; color: #15803d; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-q2 { background-color: #fee2e2; color: #b91c1c; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-q3 { background-color: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-q4 { background-color: #ccfbf1; color: #0f766e; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_detector(config: AnalysisConfig) -> EmotionDetector:
    return EmotionDetector(config)


@st.cache_resource
def get_video_processor(config: AnalysisConfig) -> VideoProcessor:
    detector = get_detector(config)
    return VideoProcessor(detector=detector, config=config)


def init_session_state():
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "session_meta" not in st.session_state:
        st.session_state.session_meta = {
            "session_title": "Academic Interview Session",
            "subject_id": "P-101",
            "interviewer": "Academic Researcher",
        }
    if "aligned_questions" not in st.session_state:
        st.session_state.aligned_questions = []
    if "live_records" not in st.session_state:
        st.session_state.live_records = []


init_session_state()

# ----------------- SIDEBAR CONTROLS -----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/psychotherapy.png", width=64)
    st.title("AffectLens Academic")
    st.caption("Multimodal Affective Analysis for Qualitative & Behavioral Research")
    st.divider()

    st.subheader("📋 Session Metadata")
    subject_id = st.text_input("Participant / Subject ID", value=st.session_state.session_meta["subject_id"])
    study_title = st.text_input("Study / Session Title", value=st.session_state.session_meta["session_title"])
    st.session_state.session_meta["subject_id"] = subject_id
    st.session_state.session_meta["session_title"] = study_title

    st.divider()
    st.subheader("⚙️ Analysis Parameters")
    sampling_fps = st.slider("Sampling Rate (FPS)", min_value=0.5, max_value=4.0, value=1.5, step=0.5,
                             help="Higher FPS provides more granular temporal resolution; lower FPS analyzes faster.")
    smoothing_w = st.slider("Smoothing Window (Frames)", min_value=1, max_value=7, value=3, step=2,
                            help="Moving average window to filter camera noise while preserving micro-reactions.")
    spike_thresh = st.slider("Emotional Spike Threshold (Δ)", min_value=0.15, max_value=0.60, value=0.30, step=0.05,
                             help="Minimum delta change in arousal or negative valence to trigger an alert.")

    config = AnalysisConfig(
        sampling_fps=sampling_fps,
        smoothing_window=smoothing_w,
        spike_threshold=spike_thresh,
    )

    st.divider()
    st.subheader("🧪 Rapid Benchmark & Demo")
    if st.button("✨ Load Standardized Academic Demo", use_container_width=True):
        with st.spinner("Generating standardized academic interview fixture..."):
            mock_timeline = generate_mock_interview_timeline(duration_sec=30.0)
            mock_summary = summarize_session_metrics(mock_timeline)
            st.session_state.analysis_result = {
                "timeline": mock_timeline,
                "summary": mock_summary,
                "metadata": {
                    "session_title": "Benchmark Interview Session",
                    "subject_id": subject_id,
                    "resolution": "1920x1080",
                    "fps": 30.0,
                    "duration_sec": 30.0,
                },
                "key_frames": [],
            }
            # Pre-load sample questions
            analyzer = MultimodalSessionAnalyzer()
            sample_questions = [
                {
                    "id": "Q1",
                    "question": "Can you describe your initial introduction to the team?",
                    "start_sec": 0.0,
                    "end_sec": 7.0,
                    "verbal_response": "I was very calm and looked forward to the new responsibilities.",
                },
                {
                    "id": "Q2",
                    "question": "How did you react when the major budget crisis hit?",
                    "start_sec": 8.0,
                    "end_sec": 17.0,
                    "verbal_response": "I claimed I had everything under control, but honestly it was pure panic.",
                },
                {
                    "id": "Q3",
                    "question": "Looking back, what was your overall reflection?",
                    "start_sec": 18.0,
                    "end_sec": 29.0,
                    "verbal_response": "We managed to overcome it together, and I felt tremendous relief and pride.",
                },
            ]
            st.session_state.aligned_questions = analyzer.align_questions(mock_timeline, sample_questions)
            st.success("Loaded academic demo dataset!")

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-header">AffectLens: Academic Interview Affect Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Scientific quantification of interviewee emotional states, Russell Circumplex dynamics, and verbal congruence.</div>', unsafe_allow_html=True)

# Tabs
tab_video, tab_multimodal, tab_live, tab_reports, tab_methodology = st.tabs([
    "🎥 Video Affect Analysis",
    "🎙️ Multimodal & Congruence Studio",
    "📹 Live Interview Monitor",
    "📊 Academic Datasets & Export",
    "📖 Scientific Methodology",
])

# ==============================================================================
# TAB 1: VIDEO AFFECT ANALYSIS
# ==============================================================================
with tab_video:
    st.subheader("1. Ingest & Analyze Interview Recording")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_video = st.file_uploader(
            "Upload Interview Video (MP4, MOV, AVI, WebM)",
            type=["mp4", "mov", "avi", "webm"],
            help="High-definition frontal view of the interviewee provides optimal scientific accuracy.",
        )

    with col_info:
        st.info(
            "**Research Tip:** For optimal facial affect coding, ensure the interviewee's face is well-lit and "
            "occupies at least 20% of the frame height."
        )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        col_btn, _ = st.columns([1, 2])
        with col_btn:
            run_btn = st.button("🚀 Run Affect Analysis", type="primary", use_container_width=True)

        if run_btn:
            progress_bar = st.progress(0, text="Initializing affect recognition pipeline...")
            status_text = st.empty()

            processor = get_video_processor(config)

            def on_progress(ratio, sec):
                pct = int(ratio * 100)
                progress_bar.progress(ratio, text=f"Analyzing interviewee reactions: {pct}% ({sec:.1f}s)")

            try:
                result = processor.process_video(
                    video_path,
                    sampling_fps=sampling_fps,
                    progress_callback=on_progress,
                    extract_key_frames=True,
                )
                progress_bar.progress(1.0, text="Analysis Complete!")
                st.session_state.analysis_result = {
                    "timeline": result.timeline,
                    "summary": result.summary,
                    "metadata": {
                        "session_title": study_title,
                        "subject_id": subject_id,
                        "resolution": result.metadata["resolution"],
                        "fps": result.metadata["fps"],
                        "duration_sec": result.metadata["duration_sec"],
                    },
                    "key_frames": result.key_frame_samples,
                }
                st.success("Interview session processed and validated!")
            except Exception as e:
                st.error(f"Error during video processing: {e}")

    # Display Dashboard if analysis results exist
    res_data = st.session_state.analysis_result
    if res_data is not None:
        summary = res_data["summary"]
        timeline = res_data["timeline"]
        meta = res_data["metadata"]

        st.divider()
        st.subheader("2. Academic Affect Summary Cards")

        kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
        with kpi1:
            dom = summary["dominant_emotion"].upper()
            color = EMOTION_COLORS.get(summary["dominant_emotion"], "#3498db")
            st.metric("Dominant Affect", dom)
        with kpi2:
            val = summary["mean_valence"]
            st.metric("Mean Valence (V)", f"{val:+.2f}", help="Pleasure vs Displeasure scale [-1.0, +1.0]")
        with kpi3:
            aro = summary["mean_arousal"]
            st.metric("Mean Arousal (A)", f"{aro:.2f}", help="Physiological activation scale [0.0, 1.0]")
        with kpi4:
            vol = summary["volatility"]["volatility_score"]
            st.metric("Volatility (EVI)", f"{vol:.1f}/100", help="Affective volatility & lability index")
        with kpi5:
            ret = summary["face_detection_rate"] * 100.0
            st.metric("Face Retention", f"{ret:.1f}%", help="Percentage of interview where face was tracked")
        with kpi6:
            spikes = summary["spikes_count"]
            st.metric("Affective Spikes", f"{spikes}", help="Moments of sudden arousal surge or stress")

        st.divider()
        st.subheader("3. Affect Dynamics & Visualizations")

        df_tl = pd.DataFrame(timeline)

        # Plot 1: Affect Timeline (Valence & Arousal)
        col_plot1, col_plot2 = st.columns([3, 2])

        with col_plot1:
            st.markdown("##### Figure 1: Continuous Affect Trajectory (Valence & Arousal)")
            fig_va = go.Figure()
            fig_va.add_trace(go.Scatter(
                x=df_tl["timestamp"],
                y=df_tl["valence"],
                mode="lines",
                name="Valence (Pleasure)",
                line=dict(color="#2563eb", width=2.5),
            ))
            fig_va.add_trace(go.Scatter(
                x=df_tl["timestamp"],
                y=df_tl["arousal"],
                mode="lines",
                name="Arousal (Activation)",
                line=dict(color="#dc2626", width=2.0, dash="dot"),
            ))

            # Add zero line for valence
            fig_va.add_hline(y=0.0, line_dash="dash", line_color="gray", opacity=0.5)

            # Highlight spikes
            for sp in summary.get("spikes", []):
                fig_va.add_vline(
                    x=sp["timestamp"],
                    line_width=1.5,
                    line_dash="dash",
                    line_color="#f59e0b",
                    annotation_text=f"Spike: {sp['trigger_dominant_emotion']}",
                    annotation_position="top left",
                )

            fig_va.update_layout(
                xaxis_title="Session Elapsed Time (seconds)",
                yaxis_title="Normalized Continuous Score",
                yaxis=dict(range=[-1.05, 1.05]),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=380,
            )
            st.plotly_chart(fig_va, use_container_width=True)

        with col_plot2:
            st.markdown("##### Figure 2: Russell's Circumplex Affect Plane")
            fig_cir = px.scatter(
                df_tl,
                x="valence",
                y="arousal",
                color="dominant_emotion",
                color_discrete_map=EMOTION_COLORS,
                hover_data=["timestamp", "quadrant_name"],
                range_x=[-1.1, 1.1],
                range_y=[-0.05, 1.05],
            )
            # Quadrant boundary guides
            fig_cir.add_vline(x=0.0, line_dash="dash", line_color="rgba(100,100,100,0.4)")
            fig_cir.add_hline(y=0.45, line_dash="dash", line_color="rgba(100,100,100,0.4)")

            fig_cir.add_annotation(x=0.6, y=0.9, text="<b>Q1: Excited/Joyful</b>", showarrow=False, font=dict(color="#15803d", size=10))
            fig_cir.add_annotation(x=-0.6, y=0.9, text="<b>Q2: Stressed/Anxious</b>", showarrow=False, font=dict(color="#b91c1c", size=10))
            fig_cir.add_annotation(x=-0.6, y=0.15, text="<b>Q3: Depressed/Fatigued</b>", showarrow=False, font=dict(color="#0369a1", size=10))
            fig_cir.add_annotation(x=0.6, y=0.15, text="<b>Q4: Calm/Serene</b>", showarrow=False, font=dict(color="#0f766e", size=10))

            fig_cir.update_layout(
                xaxis_title="Valence (Pleasure vs Displeasure)",
                yaxis_title="Arousal (Physiological Activation)",
                margin=dict(l=20, r=20, t=30, b=20),
                height=380,
            )
            st.plotly_chart(fig_cir, use_container_width=True)

        # Plot 3 & 4: Ekman 7 Emotions Distribution & Radar Chart
        col_ek1, col_ek2 = st.columns([3, 2])
        with col_ek1:
            st.markdown("##### Figure 3: Ekman 7 Discrete Emotions Over Time")
            fig_ek = go.Figure()
            for emo in EKMAN_EMOTIONS:
                prob_series = [r["emotions"].get(emo, 0.0) for r in timeline]
                fig_ek.add_trace(go.Scatter(
                    x=df_tl["timestamp"],
                    y=prob_series,
                    mode="lines",
                    name=emo.capitalize(),
                    line=dict(color=EMOTION_COLORS.get(emo, "#7f8c8d"), width=1.8),
                    stackgroup="one" if emo != "neutral" else None,
                ))
            fig_ek.update_layout(
                xaxis_title="Time (seconds)",
                yaxis_title="Emotion Probability Density",
                yaxis=dict(range=[0, 1.05]),
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=340,
            )
            st.plotly_chart(fig_ek, use_container_width=True)

        with col_ek2:
            st.markdown("##### Figure 4: Overall Affective Radar Profile")
            props = summary["emotion_proportions"]
            radar_categories = [e.capitalize() for e in EKMAN_EMOTIONS]
            radar_values = [props.get(e, 0.0) * 100.0 for e in EKMAN_EMOTIONS]
            # Close the polygon loop
            radar_categories.append(radar_categories[0])
            radar_values.append(radar_values[0])

            fig_rad = go.Figure(go.Scatterpolar(
                r=radar_values,
                theta=radar_categories,
                fill="toself",
                line=dict(color="#3b82f6", width=2),
                fillcolor="rgba(59, 130, 246, 0.25)",
            ))
            fig_rad.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(radar_values) + 10])),
                showlegend=False,
                margin=dict(l=30, r=30, t=30, b=30),
                height=340,
            )
            st.plotly_chart(fig_rad, use_container_width=True)

        # Keyframe Gallery
        if res_data.get("key_frames"):
            st.divider()
            st.subheader("4. Critical Affect Keyframe Gallery")
            st.caption("Visual inspection of interviewee facial expression during detected peaks and shifts:")
            cols = st.columns(min(len(res_data["key_frames"]), 4))
            for idx, kf in enumerate(res_data["key_frames"][:4]):
                with cols[idx]:
                    st.image(
                        kf["frame_rgb"],
                        caption=f"T: {kf['timestamp']:.1f}s | {kf['dominant_emotion'].upper()} (Val: {kf['valence']:+.2f})",
                        use_container_width=True,
                    )
    else:
        st.info("💡 To begin, upload an interview video above or click **'Load Standardized Academic Demo'** in the left sidebar.")

# ==============================================================================
# TAB 2: MULTIMODAL & CONGRUENCE STUDIO
# ==============================================================================
with tab_multimodal:
    st.subheader("Multimodal Verbal & Facial Emotion Congruence")
    st.markdown(
        "Align interview questions and verbal transcripts with facial affect to detect "
        "**emotional masking**, **cognitive dissonance**, and **non-verbal leakage**."
    )

    if st.session_state.analysis_result is None:
        st.warning("Please analyze a video or load the academic demo first to activate multimodal alignment.")
    else:
        timeline = st.session_state.analysis_result["timeline"]

        st.markdown("#### Interview Question Protocol")
        st.caption("Define the question intervals and participant verbal responses to calculate congruence:")

        q_input_method = st.radio("Input Source", ["Standard 3-Stage Interview Protocol", "Custom Question Entry"], horizontal=True)

        questions = []
        if q_input_method == "Standard 3-Stage Interview Protocol":
            max_t = timeline[-1]["timestamp"] if timeline else 30.0
            t1 = round(max_t * 0.3, 1)
            t2 = round(max_t * 0.65, 1)

            questions = [
                {
                    "id": "Q1",
                    "question": "Phase 1: Initial Rapport & Background",
                    "start_sec": 0.0,
                    "end_sec": t1,
                    "verbal_response": "I felt very prepared and comfortable coming into this discussion today.",
                },
                {
                    "id": "Q2",
                    "question": "Phase 2: Stress / Critical Challenge Inquest",
                    "start_sec": t1,
                    "end_sec": t2,
                    "verbal_response": "I told everyone I was fine and had everything handled, but inside I was trembling.",
                },
                {
                    "id": "Q3",
                    "question": "Phase 3: Resolution & Future Perspective",
                    "start_sec": t2,
                    "end_sec": max_t,
                    "verbal_response": "We solved the issue collaboratively, and I am very satisfied with our success.",
                },
            ]
        else:
            # Custom input form
            col_q1, col_q2, col_q3 = st.columns([1, 2, 2])
            with col_q1:
                cq_id = st.text_input("Question ID", "Q1")
                cq_start = st.number_input("Start Sec", value=0.0, step=1.0)
                cq_end = st.number_input("End Sec", value=10.0, step=1.0)
            with col_q2:
                cq_text = st.text_area("Question Topic", "Tell me about your experience.")
            with col_q3:
                cq_verb = st.text_area("Participant Verbal Answer", "It was good.")

            questions = [{
                "id": cq_id,
                "question": cq_text,
                "start_sec": cq_start,
                "end_sec": cq_end,
                "verbal_response": cq_verb,
            }]

        align_btn = st.button("⚖️ Calculate Multimodal Congruence Matrix", type="primary")
        if align_btn or not st.session_state.aligned_questions:
            analyzer = MultimodalSessionAnalyzer()
            st.session_state.aligned_questions = analyzer.align_questions(timeline, questions)

        aligned_res = st.session_state.aligned_questions
        if aligned_res:
            st.divider()
            st.subheader("Question-by-Question Reaction Matrix")

            rows = []
            for item in aligned_res:
                rows.append({
                    "Question": item.topic_or_question,
                    "Facial Emotion": item.dominant_facial_emotion.upper(),
                    "Facial Valence": f"{item.mean_facial_valence:+.2f}",
                    "Verbal Sentiment": f"{item.verbal_sentiment:+.2f} ({item.verbal_label})",
                    "Arousal Spike": "⚠️ Yes" if item.arousal_spike_detected else "No",
                    "Congruence Status": "✅ Congruent" if item.is_congruent else "❌ Incongruent",
                    "Scientific Assessment": item.congruence_interpretation,
                })
            df_aligned = pd.DataFrame(rows)
            st.dataframe(df_aligned, use_container_width=True)

            # Chart comparing verbal vs non-verbal sentiment
            st.markdown("##### Comparative Chart: Verbal Polarity vs Facial Valence")
            q_ids = [item.topic_or_question for item in aligned_res]
            facial_vals = [item.mean_facial_valence for item in aligned_res]
            verbal_vals = [item.verbal_sentiment for item in aligned_res]

            fig_comp = go.Figure(data=[
                go.Bar(name="Facial Valence (Non-Verbal)", x=q_ids, y=facial_vals, marker_color="#3b82f6"),
                go.Bar(name="Verbal Sentiment (NLTK VADER)", x=q_ids, y=verbal_vals, marker_color="#10b981"),
            ])
            fig_comp.update_layout(
                barmode="group",
                yaxis_title="Affect Score [-1.0, +1.0]",
                yaxis=dict(range=[-1.05, 1.05]),
                margin=dict(l=20, r=20, t=30, b=20),
                height=350,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

# ==============================================================================
# TAB 3: LIVE INTERVIEW MONITOR
# ==============================================================================
with tab_live:
    st.subheader("Live Interview Session Telemetry")
    st.caption("Capture real-time snapshots during an ongoing interview to log participant reaction dynamically.")

    col_cam, col_live_metric = st.columns([1, 1])

    with col_cam:
        cam_image = st.camera_input("Take Live Interview Frame Snapshot")

    with col_live_metric:
        if cam_image is not None:
            bytes_data = cam_image.getvalue()
            cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            detector = get_detector(config)
            det = detector.detect(cv_img, is_bgr=True, annotate=True)

            current_t = len(st.session_state.live_records) * 2.0
            live_rec = {
                "timestamp": current_t,
                "face_detected": det.face_detected,
                "dominant_emotion": det.dominant_emotion,
                "dominant_confidence": det.dominant_confidence,
                "valence": det.valence,
                "arousal": det.arousal,
                "quadrant": det.quadrant,
                "quadrant_name": det.quadrant_name,
                "emotions": det.emotions,
            }
            st.session_state.live_records.append(live_rec)

            st.success(f"Snapshot analyzed at T = {current_t:.1f}s")
            st.write(f"**Dominant State:** `{det.dominant_emotion.upper()}` (Confidence: `{det.dominant_confidence*100:.1f}%`)")
            st.write(f"**Continuous Valence:** `{det.valence:+.2f}` | **Arousal:** `{det.arousal:.2f}`")
            st.write(f"**Circumplex State:** `{det.quadrant_name}`")

            if det.annotated_frame is not None:
                rgb_annot = cv2.cvtColor(det.annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb_annot, caption="Live Affect Annotated Bounding Box", use_container_width=True)
        else:
            st.info("Capture a frame using your webcam to record live telemetry.")

    if st.session_state.live_records:
        st.divider()
        st.markdown(f"##### Live Session Log ({len(st.session_state.live_records)} frames recorded)")
        df_live = pd.DataFrame(st.session_state.live_records)
        st.dataframe(df_live[["timestamp", "dominant_emotion", "valence", "arousal", "quadrant_name"]], use_container_width=True)

        col_clr, col_adopt = st.columns([1, 1])
        with col_clr:
            if st.button("🗑️ Clear Live Session Data"):
                st.session_state.live_records = []
                st.rerun()
        with col_adopt:
            if st.button("📥 Adopt Live Log as Main Analysis Dataset"):
                live_summary = summarize_session_metrics(st.session_state.live_records)
                st.session_state.analysis_result = {
                    "timeline": st.session_state.live_records,
                    "summary": live_summary,
                    "metadata": {
                        "session_title": "Live Interview Session",
                        "subject_id": subject_id,
                        "duration_sec": live_summary["duration_seconds"],
                    },
                    "key_frames": [],
                }
                st.success("Adopted live data into primary analysis dashboard!")

# ==============================================================================
# TAB 4: ACADEMIC DATASETS & EXPORT
# ==============================================================================
with tab_reports:
    st.subheader("Academic Research Dataset & Report Exporter")
    st.markdown("Download full research schemas, statistical CSV tables, and publication-ready Markdown/LaTeX summaries.")

    if st.session_state.analysis_result is None:
        st.warning("Please analyze an interview session or load the demo dataset to export reports.")
    else:
        res = st.session_state.analysis_result
        timeline = res["timeline"]
        summary = res["summary"]
        meta = res["metadata"]
        q_sums = st.session_state.aligned_questions

        # 1. CSV
        df_csv = AcademicReportGenerator.generate_csv_dataframe(timeline)
        csv_buffer = io.StringIO()
        df_csv.to_csv(csv_buffer, index=False)
        csv_str = csv_buffer.getvalue()

        # 2. JSON
        full_json = {
            "research_session_metadata": meta,
            "scientific_summary": summary,
            "question_level_affect": [
                {
                    "question_id": q.question_id,
                    "topic": q.topic_or_question,
                    "start_sec": q.timestamp_start,
                    "end_sec": q.timestamp_end,
                    "mean_facial_valence": q.mean_facial_valence,
                    "mean_facial_arousal": q.mean_facial_arousal,
                    "dominant_emotion": q.dominant_facial_emotion,
                    "verbal_sentiment": q.verbal_sentiment,
                    "is_congruent": q.is_congruent,
                    "interpretation": q.congruence_interpretation,
                }
                for q in q_sums
            ],
            "frame_level_timeline": timeline,
        }
        json_str = json.dumps(full_json, indent=2, ensure_ascii=False)

        # 3. Markdown Report
        md_report = AcademicReportGenerator.generate_markdown_report(meta, summary, q_sums)

        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            st.download_button(
                label="📥 Download Raw CSV Dataset",
                data=csv_str,
                file_name=f"{subject_id}_affect_timeline.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c_exp2:
            st.download_button(
                label="📥 Download Scientific JSON",
                data=json_str,
                file_name=f"{subject_id}_research_data.json",
                mime="application/json",
                use_container_width=True,
            )
        with c_exp3:
            st.download_button(
                label="📥 Download Academic Report (MD)",
                data=md_report,
                file_name=f"{subject_id}_executive_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.divider()
        st.markdown("#### Scientific Report Preview")
        st.markdown(md_report)

# ==============================================================================
# TAB 5: SCIENTIFIC METHODOLOGY
# ==============================================================================
with tab_methodology:
    st.subheader("Academic Research Methodology & Formulas")
    st.markdown("""
### 1. Theoretical Framework
This framework integrates two foundational paradigms in affective computing and psychology:
1. **Paul Ekman's Discrete Basic Emotions Model (1971):**
   Quantifies expressions across 7 universal categories:
   $$\\mathcal{E} = \\{\\text{Happy}, \\text{Sad}, \\text{Angry}, \\text{Fear}, \\text{Disgust}, \\text{Surprise}, \\text{Neutral}\\}$$
2. **James Russell's Circumplex Model of Affect (1980):**
   Represents affective states on a continuous 2D plane:
   - **Valence ($V \\in [-1.0, +1.0]$):** Hedonic tone (pleasant vs. unpleasant).
   - **Arousal ($A \\in [0.0, 1.0]$):** Physiological activation (energized vs. calm).

### 2. Formulations

#### Continuous Valence:
$$V = \\sum_{e \\in \\mathcal{E}} w_v(e) \\cdot P(e)$$
where weights $w_v(\\text{happy}) = +1.0$, $w_v(\\text{surprise}) = +0.15$, $w_v(\\text{neutral}) = 0.0$, $w_v(\\text{disgust}) = -0.65$, $w_v(\\text{sad}) = -0.75$, $w_v(\\text{fear}) = -0.85$, $w_v(\\text{angry}) = -0.95$.

#### Continuous Arousal:
$$A = \\sum_{e \\in \\mathcal{E}} w_a(e) \\cdot P(e)$$
where $w_a$ maps energetic activation ($w_a(\\text{surprise}) = 0.90, w_a(\\text{fear}) = 0.85, w_a(\\text{angry}) = 0.80$, etc.).

#### Emotional Volatility Index (EVI):
Quantifies emotional lability and instability over the session:
$$\\text{EVI} = 50 \\cdot \\sigma_V + 50 \\cdot \\left( \\frac{1}{N-1} \\sum_{t=2}^N |V_t - V_{t-1}| \\right)$$
Higher EVI indicates heightened agitation, emotional swings, or vulnerability during questioning.

#### Verbal-Nonverbal Congruence:
Measures the alignment between verbal sentiment compound score $S_{\\text{text}} \\in [-1, 1]$ (NLTK VADER) and facial valence $V \\in [-1, 1]$:
$$\\Delta_{\\text{congruence}} = |V - S_{\\text{text}}|$$
A discrepancy $\\Delta > 0.40$ with conflicting signs flags **emotional suppression, masking, or cognitive dissonance**.
    """)
