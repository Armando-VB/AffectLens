#!/usr/bin/env python3
"""AffectLens Terminal MVP - Real-time Interview Facial Affect & Emotion Detection.

Objectives:
- Detect interviewee emotional state in real time using AffectNet EfficientNet-B0 ONNX.
- Apply adaptive baseline calibration to prevent resting-face bias (false anger/sadness).
- Track the exact duration of each emotional state before switching to the next.
- Display a real-time HUD panel with emotion probabilities and continuous valence waveform.
- Generate structured research artifacts: CSV records, executive summary, and visual timeline chart.
"""

import argparse
import csv
import datetime
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Standardized emotion categories
EMOTIONS_DISPLAY = {
    "happy": "HAPPY",
    "sad": "SAD",
    "angry": "ANGRY",
    "fear": "FEAR/ANXIETY",
    "disgust": "DISGUST",
    "surprise": "SURPRISE",
    "neutral": "NEUTRAL",
    "pensativo": "THINKING",
    "thinking": "THINKING",
    "sin rostro": "NO FACE",
}

# Translate any legacy Spanish labels to English
ES_TO_EN_EMOTION = {
    "FELIZ": "HAPPY",
    "TRISTE": "SAD",
    "ENOJADO": "ANGRY",
    "MIEDO/ANSIEDAD": "FEAR/ANXIETY",
    "DISGUSTO": "DISGUST",
    "SORPRESA": "SURPRISE",
    "NEUTRAL": "NEUTRAL",
    "PENSATIVO": "THINKING",
    "SIN ROSTRO": "NO FACE",
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "FEAR/ANXIETY": "FEAR/ANXIETY",
    "DISGUST": "DISGUST",
    "SURPRISE": "SURPRISE",
    "THINKING": "THINKING",
    "NO FACE": "NO FACE",
}

# BGR color palette for visualization
COLORES_BGR = {
    "HAPPY": (0, 200, 0),          # Green
    "SAD": (200, 100, 0),          # Blue
    "ANGRY": (0, 0, 220),          # Red
    "FEAR/ANXIETY": (180, 0, 180),  # Purple
    "DISGUST": (0, 140, 255),      # Orange
    "SURPRISE": (0, 220, 255),     # Yellow
    "NEUTRAL": (180, 180, 180),    # Gray
    "THINKING": (220, 180, 0),     # Cyan/Teal (Cognitive Load)
    "NO FACE": (100, 100, 100),
    # Spanish aliases for compatibility
    "FELIZ": (0, 200, 0),
    "TRISTE": (200, 100, 0),
    "ENOJADO": (0, 0, 220),
    "MIEDO/ANSIEDAD": (180, 0, 180),
    "DISGUSTO": (0, 140, 255),
    "SORPRESA": (0, 220, 255),
    "PENSATIVO": (220, 180, 0),
    "SIN ROSTRO": (100, 100, 100),
}


class EmotionTracker:
    """Tracks duration and transitions between sustained emotional states."""

    def __init__(self, umbral_cambio_frames: int = 3):
        self.umbral_frames = umbral_cambio_frames
        self.emocion_actual: Optional[str] = None
        self.tiempo_inicio_emocion: float = 0.0
        self.emocion_candidata: Optional[str] = None
        self.conteo_candidata: int = 0
        self.confianzas_acumuladas: List[float] = []

        self.historial_intervalos: List[Dict[str, Any]] = []
        self.tiempo_inicio_sesion: float = 0.0

    def iniciar(self, t_cero: float = 0.0):
        self.tiempo_inicio_sesion = t_cero
        self.tiempo_inicio_emocion = t_cero

    def actualizar(
        self,
        nueva_emocion: str,
        confianza: float,
        tiempo_actual: float,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Updates tracker with the current frame's detected emotion.

        Returns the completed interval when a stable state transition occurs.
        """
        ahora = tiempo_actual
        norm_emocion = nueva_emocion

        if self.emocion_actual is None:
            self.emocion_actual = norm_emocion
            self.tiempo_inicio_emocion = ahora
            self.confianzas_acumuladas = [confianza]
            return None

        self.confianzas_acumuladas.append(confianza)

        if norm_emocion != self.emocion_actual:
            if norm_emocion == self.emocion_candidata:
                self.conteo_candidata += 1
            else:
                self.emocion_candidata = norm_emocion
                self.conteo_candidata = 1

            # Transition confirmed after persisting across threshold frames
            if self.conteo_candidata >= self.umbral_frames:
                duracion = round(ahora - self.tiempo_inicio_emocion, 2)
                t_rel_inicio = round(self.tiempo_inicio_emocion - self.tiempo_inicio_sesion, 2)
                t_rel_fin = round(ahora - self.tiempo_inicio_sesion, 2)
                conf_media = (
                    round(float(np.mean(self.confianzas_acumuladas)), 2)
                    if self.confianzas_acumuladas
                    else confianza
                )

                intervalo_completado = {
                    "emocion": self.emocion_actual,
                    "inicio_seg": t_rel_inicio,
                    "fin_seg": t_rel_fin,
                    "duracion_seg": duracion,
                    "confianza": conf_media,
                }
                self.historial_intervalos.append(intervalo_completado)

                self.emocion_actual = norm_emocion
                self.tiempo_inicio_emocion = ahora
                self.emocion_candidata = None
                self.conteo_candidata = 0
                self.confianzas_acumuladas = [confianza]

                return intervalo_completado
        else:
            self.emocion_candidata = None
            self.conteo_candidata = 0

        return None

    def cerrar(self, tiempo_final: float) -> List[Dict[str, Any]]:
        """Finalizes and records the last active state upon session completion."""
        ahora = tiempo_final
        if self.emocion_actual is not None:
            duracion = round(ahora - self.tiempo_inicio_emocion, 2)
            if duracion > 0.1:
                t_rel_inicio = round(self.tiempo_inicio_emocion - self.tiempo_inicio_sesion, 2)
                t_rel_fin = round(ahora - self.tiempo_inicio_sesion, 2)
                conf_media = (
                    round(float(np.mean(self.confianzas_acumuladas)), 2)
                    if self.confianzas_acumuladas
                    else 0.5
                )
                self.historial_intervalos.append({
                    "emocion": self.emocion_actual,
                    "inicio_seg": t_rel_inicio,
                    "fin_seg": t_rel_fin,
                    "duracion_seg": duracion,
                    "confianza": conf_media,
                })
        return self.historial_intervalos


# Backwards compatibility alias
TrackerEmociones = EmotionTracker


def format_seconds(seg: float) -> str:
    """Formats seconds into MM:SS format."""
    m = int(seg // 60)
    s = int(seg % 60)
    return f"{m:02d}:{s:02d}"


formato_segundos = format_seconds


def save_results(
    historial: List[Dict[str, Any]],
    duracion_total: float,
    archivo_csv: str = "emotion_records.csv",
    archivo_txt: str = "session_summary.txt",
    marcas: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
):
    """Saves chronological emotion intervals and academic summary report to CSV and TXT."""
    marcas = marcas or []

    # 1. Save CSV
    with open(archivo_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["order", "start_sec", "end_sec", "duration_sec", "emotion", "avg_confidence"])
        for idx, it in enumerate(historial, 1):
            writer.writerow([
                idx,
                it["inicio_seg"],
                it["fin_seg"],
                it["duracion_seg"],
                it["emocion"],
                it["confianza"],
            ])

    # 2. Compute aggregate time per emotion
    tiempos_emocion: Dict[str, float] = {}
    for it in historial:
        emo = it["emocion"]
        tiempos_emocion[emo] = tiempos_emocion.get(emo, 0.0) + it["duracion_seg"]

    dominant_emo = "NEUTRAL"
    if tiempos_emocion:
        dominant_emo = max(tiempos_emocion, key=tiempos_emocion.get)

    # 3. Save Summary Report TXT
    lines = [
        "=" * 70,
        "   ACADEMIC RESEARCH: INTERVIEWEE EMOTION & REACTION REPORT",
        "=" * 70,
        f"Session Timestamp:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Session Duration:  {duracion_total:.1f} seconds ({duracion_total/60.0:.2f} min)",
        f"Emotion Transitions:     {len(historial)} state changes recorded",
        f"Dominant Emotion:        {dominant_emo} ({tiempos_emocion.get(dominant_emo, 0.0)/max(duracion_total, 0.01)*100:.1f}% of session)",
    ]
    if marcas:
        lines.append(f"Question Bookmarks:      {len(marcas)} event markers recorded")

    lines.extend([
        "-" * 70,
        "1. EMOTION DURATION DISTRIBUTION:",
    ])
    for emo, segs in sorted(tiempos_emocion.items(), key=lambda x: x[1], reverse=True):
        pct = (segs / duracion_total * 100.0) if duracion_total > 0 else 0.0
        lines.append(f"  - {emo:<18}: {segs:5.1f} s  ({pct:5.1f}%)")

    section_num = 2
    if marcas:
        lines.extend([
            "-" * 70,
            f"{section_num}. QUESTION / EVENT BOOKMARKS:",
        ])
        for m in marcas:
            t_m = format_seconds(m["tiempo_seg"])
            emo_val = m.get("emocion_en_marca", "-")
            lines.append(f"  • [Marker #{m['numero']} at {t_m}] Active Affect: {emo_val}")
        section_num += 1

    lines.extend([
        "-" * 70,
        f"{section_num}. CHRONOLOGICAL EMOTION TIMELINE (STATE SEQUENCING):",
    ])
    for idx, it in enumerate(historial, 1):
        ini = format_seconds(it["inicio_seg"])
        fin = format_seconds(it["fin_seg"])
        lines.append(
            f"  #{idx:02d} [{ini} -> {fin}] {it['emocion']:<16} (lasted {it['duracion_seg']:4.1f}s | conf: {int(it['confianza']*100)}%)"
        )
    lines.append("=" * 70)

    with open(archivo_txt, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


guardar_resultados = save_results
guardar_reporte_ingles = save_results


def save_timeline_chart(
    historial: List[Dict[str, Any]],
    duracion_total: float,
    archivo_png: str = "session_timeline_chart.png",
    marcas: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
):
    """Exports a publication-grade timeline chart using matplotlib."""
    if not historial:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    marcas = marcas or []

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)

    color_map = {
        "HAPPY": "#2ecc71",
        "NEUTRAL": "#95a5a6",
        "THINKING": "#1abc9c",
        "SURPRISE": "#f1c40f",
        "SAD": "#3498db",
        "FEAR/ANXIETY": "#9b59b6",
        "ANGRY": "#e74c3c",
        "DISGUST": "#e67e22",
        "NO FACE": "#7f8c8d",
    }

    y_labels = list(color_map.keys())
    for it in historial:
        emo = ES_TO_EN_EMOTION.get(it["emocion"], it["emocion"].upper())
        c = color_map.get(emo, "#333333")
        ax.barh(
            emo,
            width=it["duracion_seg"],
            left=it["inicio_seg"],
            height=0.6,
            color=c,
            edgecolor="white",
            alpha=0.9,
        )

    # Question bookmarks
    for m in marcas:
        t = m["tiempo_seg"]
        ax.axvline(x=t, color="#e67e22", linestyle="--", linewidth=1.5, alpha=0.85)
        ax.text(
            t,
            len(y_labels) - 0.4,
            f"Q{m['numero']}",
            color="#d35400",
            fontweight="bold",
            fontsize=9,
            ha="center",
        )

    ax.set_xlabel("Session Elapsed Time (seconds)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Affective State", fontsize=10, fontweight="bold")
    ax.set_title("Interviewee Emotion Evolution Timeline (Academic Research)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(duracion_total, 1.0))
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()

    plt.savefig(archivo_png, bbox_inches="tight")
    plt.close()


guardar_grafico_resumen = save_timeline_chart


def render_realtime_analytics_panel(
    alto: int,
    ancho: int,
    emocion_actual: str,
    duracion_actual: float,
    emociones_dict: Dict[str, float],
    buffer_valencia: List[float],
    **kwargs,
) -> np.ndarray:
    """Renders a sleek real-time analytics panel alongside the camera feed."""
    panel = np.full((alto, ancho, 3), 24, dtype=np.uint8)

    # 1. Header
    cv2.putText(
        panel,
        "AFFECTLENS LIVE ANALYTICS",
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # 2. Emotion State Badge
    norm_emo = ES_TO_EN_EMOTION.get(emocion_actual, emocion_actual.upper())
    color_emo = COLORES_BGR.get(norm_emo, (200, 200, 200))
    cv2.rectangle(panel, (15, 34), (ancho - 15, 74), (38, 38, 38), -1)
    cv2.rectangle(panel, (15, 34), (ancho - 15, 74), color_emo, 1)
    cv2.putText(
        panel,
        f"AFFECT: {norm_emo}",
        (22, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color_emo,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{duracion_actual:.1f}s",
        (ancho - 62, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 3. Live Probability Bars
    cv2.putText(
        panel,
        "EMOTION PROBABILITIES",
        (15, 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    orden_emos = [
        ("happy", "HAPPY"),
        ("neutral", "NEUTRAL"),
        ("pensativo", "THINKING"),
        ("surprise", "SURPRISE"),
        ("sad", "SAD"),
        ("fear", "FEAR"),
        ("angry", "ANGRY"),
        ("disgust", "DISGUST"),
    ]

    y_start = 112
    max_bar_w = ancho - 145
    for idx, (k, nombre) in enumerate(orden_emos):
        prob = emociones_dict.get(k, 0.0)
        bar_w = int(prob * max_bar_w)
        bar_y = y_start + idx * 22
        col_bar = COLORES_BGR.get(nombre, (180, 180, 180))

        # Emotion Name
        cv2.putText(
            panel,
            f"{nombre[:8]:<8}",
            (15, bar_y + 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        # Background slot
        cv2.rectangle(panel, (80, bar_y), (80 + max_bar_w, bar_y + 12), (44, 44, 44), -1)
        # Dynamic filled bar
        if bar_w > 0:
            cv2.rectangle(panel, (80, bar_y), (80 + bar_w, bar_y + 12), col_bar, -1)
        # Numeric percentage
        cv2.putText(
            panel,
            f"{int(prob * 100)}%",
            (80 + max_bar_w + 6, bar_y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

    # 4. Real-Time Valence Trajectory Waveform Graph
    graph_y_top = y_start + len(orden_emos) * 22 + 22
    graph_h = alto - graph_y_top - 38
    if graph_h > 45:
        cv2.putText(
            panel,
            "VALENCE TRAJECTORY (EVOLUTION)",
            (15, graph_y_top - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        gx1, gx2 = 15, ancho - 15
        gy1, gy2 = graph_y_top, graph_y_top + graph_h
        cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (36, 36, 36), -1)
        cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (60, 60, 60), 1)

        # Baseline zero line
        mid_y = gy1 + graph_h // 2
        cv2.line(panel, (gx1, mid_y), (gx2, mid_y), (80, 80, 80), 1)
        cv2.putText(panel, "+ Joy", (gx1 + 4, gy1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 220, 100), 1, cv2.LINE_AA)
        cv2.putText(panel, "- Stress", (gx1 + 4, gy2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 240), 1, cv2.LINE_AA)

        if len(buffer_valencia) >= 2:
            gw = gx2 - gx1
            step = gw / max(len(buffer_valencia) - 1, 1)

            pts = []
            for i, v in enumerate(buffer_valencia):
                px = int(gx1 + i * step)
                norm_v = (v + 1.0) / 2.0
                py = int(gy2 - norm_v * graph_h)
                py = int(np.clip(py, gy1, gy2))
                pts.append((px, py))

            for i in range(len(pts) - 1):
                c_line = (0, 220, 100) if buffer_valencia[i] >= 0.0 else (80, 80, 240)
                cv2.line(panel, pts[i], pts[i + 1], c_line, 2)

            last_x, last_y = pts[-1]
            cv2.circle(panel, (last_x, last_y), 4, (0, 255, 255), -1)

    # 5. Footer Shortcuts
    cv2.putText(
        panel,
        "[SPACE] Mark Q | [P] Pause | [Q] Stop",
        (15, alto - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (130, 130, 130),
        1,
        cv2.LINE_AA,
    )

    return panel


dibujar_panel_grafico_en_tiempo_real = render_realtime_analytics_panel


def print_terminal_summary(
    historial: List[Dict[str, Any]],
    duracion_total: float,
    ruta_csv: str,
    ruta_txt: str,
    ruta_chart: Optional[str] = None,
    marcas: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
):
    """Prints a clean, formatted academic summary in terminal at session end."""
    marcas = marcas or []

    print("\n" + "=" * 70)
    print("               FINAL RESEARCH SESSION SUMMARY")
    print("=" * 70)
    print(f" Total Duration:      {duracion_total:.1f} seconds ({duracion_total/60.0:.2f} min)")
    print(f" Total Transitions:   {len(historial)} emotional state changes")
    if marcas:
        print(f" Bookmarked Events:   {len(marcas)} question markers")
    print("-" * 70)
    print(" 1. EMOTION DURATION DISTRIBUTION:")

    tiempos: Dict[str, float] = {}
    for it in historial:
        emo = ES_TO_EN_EMOTION.get(it["emocion"], it["emocion"].upper())
        tiempos[emo] = tiempos.get(emo, 0.0) + it["duracion_seg"]

    for emo, segs in sorted(tiempos.items(), key=lambda x: x[1], reverse=True):
        pct = (segs / duracion_total * 100.0) if duracion_total > 0 else 0.0
        barra = "█" * int(pct // 5)
        print(f"  • {emo:<16} : {segs:5.1f}s ({pct:5.1f}%)  {barra}")

    sec_num = 2
    if marcas:
        print("-" * 70)
        print(f" {sec_num}. QUESTION / EVENT BOOKMARKS:")
        for m in marcas:
            t_m = format_seconds(m["tiempo_seg"])
            print(f"  📌 Question #{m['numero']} at {t_m} -> Active Emotion: {m.get('emocion_en_marca', '-')}")
        sec_num += 1

    print("-" * 70)
    print(f" {sec_num}. CHRONOLOGICAL TIMELINE (EMOTION & DURATION):")
    for idx, it in enumerate(historial, 1):
        ini = format_seconds(it["inicio_seg"])
        fin = format_seconds(it["fin_seg"])
        emo = ES_TO_EN_EMOTION.get(it["emocion"], it["emocion"].upper())
        print(f"  #{idx:02d} [{ini} -> {fin}]  {emo:<16} (lasted {it['duracion_seg']:4.1f}s | conf: {int(it['confianza']*100)}%)")

    print("=" * 70)
    print(" 💾 Saved Research Data:")
    print(f"    - Detailed CSV     : {Path(ruta_csv).resolve()}")
    print(f"    - Executive Summary: {Path(ruta_txt).resolve()}")
    if ruta_chart:
        print(f"    - Timeline Chart   : {Path(ruta_chart).resolve()}")
    print("=" * 70 + "\n")


imprimir_resumen_terminal = print_terminal_summary


def run_mvp(
    video_fuente: Optional[str] = None,
    cam_index: int = 0,
    mostrar_ventana: bool = True,
    csv_salida: str = "emotion_records.csv",
    txt_salida: str = "session_summary.txt",
    chart_salida: str = "session_timeline_chart.png",
    auto_start: bool = False,
    **kwargs,
):
    """Main execution loop for real-time facial affect and emotion detection."""
    print("\n" + "=" * 70)
    print("   AFFECTLENS: REAL-TIME INTERVIEW EMOTION & REACTION ANALYZER")
    print("=" * 70)
    print(" Loading AffectNet ONNX model (EfficientNet-B0) with baseline calibration...")
    from affect_analyzer.robust_detector import RobustEmotionDetector
    detector = RobustEmotionDetector()

    fuente = video_fuente if video_fuente else cam_index
    cap = cv2.VideoCapture(fuente)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera (index {cam_index}) or specified video.", file=sys.stderr)
        if video_fuente is None:
            print("👉 Try running with a sample video file:")
            print("   python main.py --video sample_data/sample_interview.mp4")
        sys.exit(1)

    print("✅ Model loaded and ready.")
    print("-" * 70)
    print(" HOW IT WORKS:")
    print(" 1. Auto-calibrates the interviewee's resting face during the first 2-3 seconds.")
    print(" 2. Displays live emotion probabilities and real-time valence trajectory.")
    print(" 3. Automatically tracks how long each emotion lasts before switching.")
    print(" 4. Exports structured CSV data, summary report, and PNG chart upon completion.")
    print("-" * 70)

    if not auto_start:
        try:
            input("▶️  Press [ENTER] to start session (PLAY)... ")
        except EOFError:
            pass

    es_archivo_video = video_fuente is not None
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0 if es_archivo_video else 30.0

    tracker = EmotionTracker(umbral_cambio_frames=2)
    tracker.iniciar(t_cero=0.0)

    t_inicio = time.time()
    t_actual_rel = 0.0

    print("\n🔴 SESSION RUNNING (Press 'q' in window or Ctrl+C in terminal to stop)")
    print("=" * 70)
    print(f"{'TIME':<10} | {'CURRENT EMOTION':<18} | {'PREVIOUS DURATION':<18}")
    print("-" * 70)

    ultimo_estado_impreso = None
    frame_count = 0

    marcas_preguntas: List[Dict[str, Any]] = []
    conteo_marcas = 0
    en_pausa = False

    buffer_valencia: List[float] = [0.0]
    emociones_dict_actual: Dict[str, float] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if es_archivo_video:
                t_actual_rel = frame_count / native_fps
            else:
                t_actual_rel = time.time() - t_inicio

            if en_pausa:
                if mostrar_ventana:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 100, 200), -1)
                    cv2.putText(
                        frame,
                        "SESSION PAUSED - Press 'p' to resume",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("AffectLens - Real-Time Emotion Analyzer", frame)
                    k = cv2.waitKey(30) & 0xFF
                    if k == ord("p") or k == ord("P"):
                        en_pausa = False
                        print("▶️  Session resumed.")
                    elif k == ord("q") or k == 27:
                        break
                continue

            # Analyze every 2 frames for optimal responsiveness
            if frame_count % 2 == 0:
                det = detector.detect(frame)

                emocion_str = ES_TO_EN_EMOTION.get(det.dominant_emotion, det.dominant_emotion.upper())
                confianza = det.confidence
                caja = det.bounding_box
                emociones_dict_actual = det.probabilities

                buffer_valencia.append(det.valence)
                if len(buffer_valencia) > 50:
                    buffer_valencia.pop(0)

                cambio = tracker.actualizar(emocion_str, confianza, tiempo_actual=t_actual_rel)
                tiempo_str = format_seconds(t_actual_rel)

                if cambio is not None:
                    ini_s = format_seconds(cambio["inicio_seg"])
                    fin_s = format_seconds(cambio["fin_seg"])
                    print(
                        f"[{ini_s}->{fin_s}] {cambio['emocion']:<18} | Lasted: {cambio['duracion_seg']:4.1f}s (Conf: {int(cambio['confianza']*100)}%)"
                    )
                    print(f"  ↳ Now transitioning to: {emocion_str}")
                elif emocion_str != ultimo_estado_impreso:
                    print(f"[{tiempo_str}]   {emocion_str:<18} | [Starting state]")
                    ultimo_estado_impreso = emocion_str

                if mostrar_ventana:
                    dur_actual = t_actual_rel - tracker.tiempo_inicio_emocion
                    color = COLORES_BGR.get(emocion_str, (255, 255, 255))

                    if caja is not None:
                        x, y, w, h = caja
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Top status bar
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (30, 30, 30), -1)
                    if not det.is_calibrated:
                        texto_estado = f"CALIBRATING BASELINE... | Session: {tiempo_str}"
                    else:
                        texto_estado = f"FEELING: {emocion_str} ({dur_actual:.1f}s) | Session: {tiempo_str}"
                    cv2.putText(
                        frame,
                        texto_estado,
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    # Side HUD panel
                    panel_w = 340
                    panel = render_realtime_analytics_panel(
                        alto=frame.shape[0],
                        ancho=panel_w,
                        emocion_actual=emocion_str,
                        duracion_actual=dur_actual,
                        emociones_dict=emociones_dict_actual,
                        buffer_valencia=buffer_valencia,
                    )

                    vista_completa = np.hstack([frame, panel])

                    try:
                        cv2.imshow("AffectLens - Real-Time Emotion Analyzer", vista_completa)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                        elif key == ord(" "):
                            conteo_marcas += 1
                            marcas_preguntas.append({
                                "numero": conteo_marcas,
                                "tiempo_seg": round(t_actual_rel, 2),
                                "emocion_en_marca": emocion_str,
                            })
                            print(f"\n📌 [{tiempo_str}] QUESTION #{conteo_marcas} BOOKMARKED (Emotion: {emocion_str})\n")
                        elif key == ord("p") or key == ord("P"):
                            en_pausa = True
                            print(f"\n⏸️  [{tiempo_str}] SESSION PAUSED. Press 'p' in window to resume.")
                    except Exception:
                        mostrar_ventana = False

    except KeyboardInterrupt:
        print("\n⏹️  Stopping session by user...")

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    duracion_total = round(t_actual_rel if es_archivo_video else (time.time() - t_inicio), 2)
    historial = tracker.cerrar(tiempo_final=duracion_total)

    # 1. Save CSV
    save_results(
        historial,
        duracion_total,
        archivo_csv=csv_salida,
        archivo_txt=txt_salida,
        marcas=marcas_preguntas,
    )

    # 2. Save Timeline Chart
    save_timeline_chart(
        historial,
        duracion_total,
        archivo_png=chart_salida,
        marcas=marcas_preguntas,
    )

    # 3. Print Terminal Summary
    print_terminal_summary(
        historial,
        duracion_total,
        csv_salida,
        txt_salida,
        ruta_chart=chart_salida,
        marcas=marcas_preguntas,
    )


ejecutar_mvp = run_mvp


def main():
    parser = argparse.ArgumentParser(
        description="AffectLens - Real-Time Interview Emotion Detection & Duration Analyzer"
    )
    parser.add_argument(
        "--video", "-v",
        help="Path to video file (optional). If omitted, live webcam is used.",
    )
    parser.add_argument(
        "--cam", "-c",
        type=int,
        default=0,
        help="Webcam device index (default: 0).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run strictly in terminal without pop-up video window.",
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start recording immediately without waiting for ENTER.",
    )
    parser.add_argument(
        "--output-csv",
        default="emotion_records.csv",
        help="Output CSV file path (default: emotion_records.csv).",
    )
    parser.add_argument(
        "--output-summary", "--output-txt",
        dest="output_summary",
        default="session_summary.txt",
        help="Output TXT summary file path (default: session_summary.txt).",
    )
    parser.add_argument(
        "--output-chart",
        default="session_timeline_chart.png",
        help="Output PNG timeline chart file path (default: session_timeline_chart.png).",
    )

    args = parser.parse_args()

    run_mvp(
        video_fuente=args.video,
        cam_index=args.cam,
        mostrar_ventana=not args.no_gui,
        csv_salida=args.output_csv,
        txt_salida=args.output_summary,
        chart_salida=args.output_chart,
        auto_start=args.auto_start,
    )


if __name__ == "__main__":
    main()
