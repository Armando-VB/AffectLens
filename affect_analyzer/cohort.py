"""Cohort Statistical Aggregator for Academic Studies.

Aggregates multiple interviewee sessions, compares experimental cohorts
(e.g., Control vs. Treatment groups), and computes inferential statistics
(Student's t-test, p-value, Cohen's d effect size, and group variance).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
from scipy import stats

from affect_analyzer.config import EKMAN_EMOTIONS


@dataclass
class GroupComparisonResult:
    """Inferential statistical comparison between two participant groups."""
    metric_name: str
    group_a_name: str
    group_b_name: str
    group_a_mean: float
    group_a_std: float
    group_b_mean: float
    group_b_std: float
    t_statistic: float
    p_value: float
    cohens_d: float
    statistically_significant: bool  # p < 0.05


class CohortAggregator:
    """Analyzes and compares multi-participant interview research cohorts."""

    @staticmethod
    def aggregate_sessions(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute descriptive statistics across a cohort of interviewees."""
        if not sessions:
            return {"sample_size": 0}

        n = len(sessions)
        valences = [s.get("summary", {}).get("mean_valence", 0.0) for s in sessions]
        arousals = [s.get("summary", {}).get("mean_arousal", 0.0) for s in sessions]
        volatilities = [
            s.get("summary", {}).get("volatility", {}).get("volatility_score", 0.0)
            for s in sessions
        ]
        face_rates = [s.get("summary", {}).get("face_detection_rate", 0.0) for s in sessions]

        # Emotion averages across cohort
        cohort_emotions: Dict[str, float] = {}
        for emo in EKMAN_EMOTIONS:
            vals = [
                s.get("summary", {}).get("emotion_proportions", {}).get(emo, 0.0)
                for s in sessions
            ]
            cohort_emotions[emo] = round(float(np.mean(vals)), 4) if vals else 0.0

        return {
            "sample_size": n,
            "mean_valence": round(float(np.mean(valences)), 4),
            "std_valence": round(float(np.std(valences, ddof=1)), 4) if n > 1 else 0.0,
            "mean_arousal": round(float(np.mean(arousals)), 4),
            "std_arousal": round(float(np.std(arousals, ddof=1)), 4) if n > 1 else 0.0,
            "mean_volatility": round(float(np.mean(volatilities)), 2),
            "mean_face_retention": round(float(np.mean(face_rates)), 4),
            "cohort_emotion_profile": cohort_emotions,
        }

    @staticmethod
    def compare_groups(
        group_a: List[Dict[str, Any]],
        group_b: List[Dict[str, Any]],
        group_a_name: str = "Group A (Control)",
        group_b_name: str = "Group B (Experimental)",
    ) -> List[GroupComparisonResult]:
        """Perform two-sample t-tests and Cohen's d effect sizes across key metrics."""
        results: List[GroupComparisonResult] = []
        metrics_to_test = [
            ("mean_valence", "Valence (Pleasure vs Displeasure)"),
            ("mean_arousal", "Arousal (Physiological Activation)"),
            ("volatility_score", "Affective Volatility (EVI)"),
        ]

        for key, display_name in metrics_to_test:
            if key == "volatility_score":
                vals_a = [s.get("summary", {}).get("volatility", {}).get("volatility_score", 0.0) for s in group_a]
                vals_b = [s.get("summary", {}).get("volatility", {}).get("volatility_score", 0.0) for s in group_b]
            else:
                vals_a = [s.get("summary", {}).get(key, 0.0) for s in group_a]
                vals_b = [s.get("summary", {}).get(key, 0.0) for s in group_b]

            mean_a = float(np.mean(vals_a)) if vals_a else 0.0
            std_a = float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else 0.0
            mean_b = float(np.mean(vals_b)) if vals_b else 0.0
            std_b = float(np.std(vals_b, ddof=1)) if len(vals_b) > 1 else 0.0

            if len(vals_a) > 1 and len(vals_b) > 1:
                t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
                # Pooled standard deviation for Cohen's d
                n1, n2 = len(vals_a), len(vals_b)
                s_pooled = np.sqrt(((n1 - 1) * (std_a**2) + (n2 - 1) * (std_b**2)) / (n1 + n2 - 2))
                cohens_d = (mean_b - mean_a) / s_pooled if s_pooled > 0 else 0.0
            else:
                t_stat, p_val, cohens_d = 0.0, 1.0, 0.0

            results.append(GroupComparisonResult(
                metric_name=display_name,
                group_a_name=group_a_name,
                group_b_name=group_b_name,
                group_a_mean=round(mean_a, 3),
                group_a_std=round(std_a, 3),
                group_b_mean=round(mean_b, 3),
                group_b_std=round(std_b, 3),
                t_statistic=round(float(t_stat), 3),
                p_value=round(float(p_val), 4),
                cohens_d=round(float(cohens_d), 3),
                statistically_significant=bool(p_val < 0.05),
            ))

        return results
