"""
moodlens_pipeline.py
=====================
MoodLens — Full Pipeline

FLOW:
  1. Per-week: load 30-item response → scale 9 features
               XGBoost predicts  → probability vector + label
               RF predicts       → probability vector + label
               Model Debate      → decide winner, confidence tier, agreement flag
               SHAP              → per-feature contributions for winning model
               Store weekly record

  2. After Week 4: aggregate all records → build report data dict
                   generate_report()    → PDF

USAGE:
  # Single week (call each week as responses come in)
  record = run_weekly_inference(raw_responses, week_number=1)

  # After week 4
  generate_report_from_records(records, output_path="report.pdf")

  # Or run the full demo with SAMPLE_RESPONSES
  python moodlens_pipeline.py
"""

# ── stdlib ───────────────────────────────────────────────────
import io
import math
import warnings
warnings.filterwarnings("ignore")

# ── third-party ──────────────────────────────────────────────
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)

# ════════════════════════════════════════════════════════════
# CONSTANTS — column definitions
# ════════════════════════════════════════════════════════════

# The 9 features the models were trained on (in order)
MODEL_FEATURES = [
    "Q1_Interested", "Q2_Distressed", "Q4_Upset", "Q7_Scared",
    "Q9_Enthusiastic", "Q11_Irritable", "Q15_Nervous",
    "Q18_Jittery", "Focus_Ability"
]

# PA items (10) — for computing PA score
PA_COLS = [
    "Q1_Interested", "Q3_Excited", "Q5_Strong", "Q9_Enthusiastic",
    "Q10_Proud", "Q12_Alert", "Q14_Inspired", "Q16_Determined",
    "Q17_Attentive", "Q19_Active"
]

# NA items (8 for PA/NA score, 10 shown in report)
NA_COLS = [
    "Q2_Distressed", "Q4_Upset", "Q7_Scared", "Q8_Hostile",
    "Q11_Irritable", "Q15_Nervous", "Q18_Jittery", "Q20_Afraid"
]

# Plutchik emotion → source columns
PLUTCHIK_MAP = {
    "Joy":          ["Q9_Enthusiastic", "Q3_Excited"],
    "Trust":        ["Q1_Interested",   "Q17_Attentive"],
    "Fear":         ["Q7_Scared",       "Q15_Nervous"],
    "Surprise":     ["Q18_Jittery"],
    "Sadness":      ["Q2_Distressed",   "Q4_Upset"],
    "Disgust":      ["Q8_Hostile",      "Q13_Ashamed"],
    "Anger":        ["Q11_Irritable",   "Q8_Hostile"],
    "Anticipation": ["Q16_Determined",  "Q14_Inspired"],
}

LABEL_MAP = {0: "Stable", 1: "Moderate", 2: "Unstable"}
LABEL_INV  = {"Stable": 0, "Moderate": 1, "Unstable": 2}


# ════════════════════════════════════════════════════════════
# SECTION A — MODEL DEBATE ENGINE
# ════════════════════════════════════════════════════════════

def scale_features(raw_responses: dict) -> np.ndarray:
    """
    Scale the 9 model features from 1-5 Likert to [0,1].
    Formula: (x - 1) / 4
    Returns a (1, 9) numpy array ready for model.predict()
    """
    scaled = [(raw_responses[f] - 1) / 4.0 for f in MODEL_FEATURES]
    return np.array(scaled).reshape(1, -1)


def model_debate(xgb_model, rf_model,
                 x_scaled: np.ndarray,
                 week_number: int) -> dict:
    """
    Run both models on x_scaled and decide the winner.

    Decision logic (3-round debate):
    ──────────────────────────────────────────────────────────
    Round 1 — Agreement check
        If both models predict the same label → UNANIMOUS.
        Winner = that label. Confidence = mean of both max probs.

    Round 2 — Confidence gap (when they disagree)
        Compute max probability (confidence) for each model.
        If gap ≥ 0.15  → higher confidence model wins.
        Winner = that model's label.

    Round 3 — Tie-break (gap < 0.15, models disagree)
        XGBoost wins by default (paper justification:
        marginally higher accuracy, tighter F1 spread,
        better probability calibration — see Section IV-A).

    Confidence tier:
        ≥ 0.80 → HIGH
        ≥ 0.60 → MEDIUM
        < 0.60 → LOW

    Returns a dict with everything needed for the report.
    ──────────────────────────────────────────────────────────
    """
    # ── predictions ──────────────────────────────────────────
    xgb_probs  = xgb_model.predict_proba(x_scaled)[0]   # shape (3,)
    rf_probs   = rf_model.predict_proba(x_scaled)[0]

    xgb_label_idx = int(np.argmax(xgb_probs))
    rf_label_idx  = int(np.argmax(rf_probs))

    xgb_label     = LABEL_MAP[xgb_label_idx]
    rf_label      = LABEL_MAP[rf_label_idx]

    xgb_confidence = float(xgb_probs[xgb_label_idx])
    rf_confidence  = float(rf_probs[rf_label_idx])

    # ── Round 1: agreement ───────────────────────────────────
    if xgb_label == rf_label:
        debate_outcome  = "UNANIMOUS"
        winner_model    = "XGBoost + RF (agreed)"
        final_label     = xgb_label
        final_label_idx = xgb_label_idx
        final_confidence = (xgb_confidence + rf_confidence) / 2
        disagreement_note = None

    else:
        # ── Round 2: confidence gap ───────────────────────────
        gap = abs(xgb_confidence - rf_confidence)

        if gap >= 0.15:
            if xgb_confidence >= rf_confidence:
                debate_outcome   = "XGBoost wins (higher confidence)"
                winner_model     = "XGBoost"
                final_label      = xgb_label
                final_label_idx  = xgb_label_idx
                final_confidence = xgb_confidence
            else:
                debate_outcome   = "RF wins (higher confidence)"
                winner_model     = "Random Forest"
                final_label      = rf_label
                final_label_idx  = rf_label_idx
                final_confidence = rf_confidence
        else:
            # ── Round 3: tie-break → XGBoost default ──────────
            debate_outcome   = "XGBoost wins (tie-break — default)"
            winner_model     = "XGBoost"
            final_label      = xgb_label
            final_label_idx  = xgb_label_idx
            final_confidence = xgb_confidence

        disagreement_note = (
            f"Models disagreed: XGBoost→{xgb_label} "
            f"({xgb_confidence:.1%}), RF→{rf_label} "
            f"({rf_confidence:.1%}). {debate_outcome}."
        )

    # ── Confidence tier ───────────────────────────────────────
    if final_confidence >= 0.80:
        confidence_tier = "HIGH"
    elif final_confidence >= 0.60:
        confidence_tier = "MEDIUM"
    else:
        confidence_tier = "LOW"

    # ── Full probability breakdown (for report display) ───────
    class_probs = {
        LABEL_MAP[i]: {
            "xgb": round(float(xgb_probs[i]), 4),
            "rf":  round(float(rf_probs[i]),  4),
        }
        for i in range(3)
    }

    return {
        "week":              week_number,
        "final_label":       final_label,
        "final_label_idx":   final_label_idx,
        "final_confidence":  round(final_confidence, 4),
        "confidence_tier":   confidence_tier,
        "winner_model":      winner_model,
        "debate_outcome":    debate_outcome,
        "disagreement_note": disagreement_note,
        "xgb_label":         xgb_label,
        "xgb_confidence":    round(xgb_confidence, 4),
        "rf_label":          rf_label,
        "rf_confidence":     round(rf_confidence, 4),
        "class_probs":       class_probs,
        "agreed":            (xgb_label == rf_label),
    }


def compute_shap(xgb_model, x_scaled: np.ndarray) -> dict:
    """
    Compute SHAP values for the primary model prediction.

    PRODUCTION (with real XGBoost + shap installed):
        import shap
        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer.shap_values(x_scaled)
        if isinstance(shap_vals, list):
            predicted_class = int(np.argmax(xgb_model.predict_proba(x_scaled)[0]))
            sv = shap_vals[predicted_class][0]
        else:
            predicted_class = int(np.argmax(xgb_model.predict_proba(x_scaled)[0]))
            sv = shap_vals[0, :, predicted_class]
        return {feat: round(float(sv[i]), 6) for i, feat in enumerate(MODEL_FEATURES)}

    DEMO fallback below — replace with the block above in production.
    """
    probs           = xgb_model.predict_proba(x_scaled)[0]
    predicted_class = int(np.argmax(probs))
    confidence      = float(probs[predicted_class])
    x_flat          = x_scaled[0]
    sv = []
    for val in x_flat:
        if predicted_class == 2:
            sv.append((val - 0.5) * confidence * 0.18)
        elif predicted_class == 0:
            sv.append((0.5 - val) * confidence * 0.18)
        else:
            sv.append((val - 0.5) * confidence * 0.08)
    return {feat: round(float(sv[i]), 6) for i, feat in enumerate(MODEL_FEATURES)}


# ════════════════════════════════════════════════════════════
# SECTION B — WEEKLY INFERENCE
# ════════════════════════════════════════════════════════════

def run_weekly_inference(raw_responses: dict,
                         week_number: int,
                         xgb_model,
                         rf_model) -> dict:
    """
    Full per-week inference pipeline.

    Parameters
    ----------
    raw_responses : dict
        All 30 questionnaire columns, Likert 1–5.
        Keys must include all MODEL_FEATURES, PA_COLS, NA_COLS, etc.
    week_number : int   (1–4)
    xgb_model, rf_model : loaded sklearn-compatible models

    Returns
    -------
    weekly_record : dict
        Everything needed to build the report for this week.
    """
    # Scale 9 features
    x_scaled = scale_features(raw_responses)

    # Model debate
    debate   = model_debate(xgb_model, rf_model, x_scaled, week_number)

    # SHAP (always on XGBoost — primary model)
    shap_vals = compute_shap(xgb_model, x_scaled)

    # PA / NA scores from scaled values
    def scaled(col):
        return (raw_responses[col] - 1) / 4.0

    pa_score = float(np.mean([scaled(c) for c in PA_COLS if c in raw_responses]))
    na_score = float(np.mean([scaled(c) for c in NA_COLS if c in raw_responses]))

    # Plutchik emotion scores (averaged across 4 weeks later, here per week)
    plutchik_week = {
        emotion: float(np.mean([scaled(c) for c in cols if c in raw_responses]))
        for emotion, cols in PLUTCHIK_MAP.items()
    }

    # Cognitive indicators
    cognitive_week = {
        "Emotional Clarity":    scaled("Q21_Emotional_Clarity") if "Q21_Emotional_Clarity" in raw_responses else None,
        "Somatic Awareness":    (scaled("Q22_Physical_Awareness") if "Q22_Physical_Awareness" in raw_responses
                                else scaled("Somatic_Awareness") if "Somatic_Awareness" in raw_responses else None),
        "Rumination":           (scaled("Q24_Rumination") if "Q24_Rumination" in raw_responses
                                else scaled("Rumination") if "Rumination" in raw_responses else None),
        "Psychological Safety": (scaled("Q25_Safety") if "Q25_Safety" in raw_responses
                                else scaled("Psychological_Safety") if "Psychological_Safety" in raw_responses else None),
        "Social Connection":    scaled("Social_Connection")      if "Social_Connection"      in raw_responses else None,
    }

    # Lifestyle factors
    lifestyle_week = {
        "Sleep Quality":    scaled("Sleep_Quality")    if "Sleep_Quality"    in raw_responses else None,
        "Stress Level":     scaled("Daily_Stress")     if "Daily_Stress"     in raw_responses else None,
        "Focus Ability":    scaled("Focus_Ability")    if "Focus_Ability"    in raw_responses else None,
        "Physical Fatigue": scaled("Physical_Fatigue") if "Physical_Fatigue" in raw_responses else None,
    }

    # PA / NA dimension detail
    pa_dims_week = {
        col.replace("Q","").split("_",1)[-1] if "_" in col else col:
        scaled(col) for col in PA_COLS if col in raw_responses
    }
    na_dims_week = {
        col.replace("Q","").split("_",1)[-1] if "_" in col else col:
        scaled(col) for col in (NA_COLS + ["Q6_Guilty","Q13_Ashamed"])
        if col in raw_responses
    }

    record = {
        "week":            week_number,
        "raw_responses":   raw_responses,
        "x_scaled":        x_scaled,
        "mood_label":      debate["final_label"],
        "pa_score":        round(pa_score, 4),
        "na_score":        round(na_score, 4),
        "debate":          debate,
        "shap":            shap_vals,
        "plutchik_week":   plutchik_week,
        "cognitive_week":  cognitive_week,
        "lifestyle_week":  lifestyle_week,
        "pa_dims_week":    pa_dims_week,
        "na_dims_week":    na_dims_week,
    }

    # Print debate summary to console each week
    _print_week_summary(record)
    return record


def _print_week_summary(record: dict):
    d = record["debate"]
    print(f"\n{'='*55}")
    print(f"  WEEK {record['week']} — MODEL DEBATE RESULT")
    print(f"{'='*55}")
    print(f"  XGBoost  → {d['xgb_label']:<10} ({d['xgb_confidence']:.1%})")
    print(f"  RF       → {d['rf_label']:<10} ({d['rf_confidence']:.1%})")
    print(f"  Outcome  : {d['debate_outcome']}")
    print(f"  ──────────────────────────────────────────────────")
    print(f"  FINAL    → {d['final_label']:<10} "
          f"[{d['confidence_tier']} confidence: {d['final_confidence']:.1%}]")
    if d["disagreement_note"]:
        print(f"  NOTE     : {d['disagreement_note']}")
    print(f"{'='*55}")
    print(f"  PA Score : {record['pa_score']:.3f}   "
          f"NA Score: {record['na_score']:.3f}   "
          f"Balance: {record['pa_score']-record['na_score']:+.3f}")
    top_shap = sorted(record["shap"].items(),
                      key=lambda x: abs(x[1]), reverse=True)[:3]
    print(f"  Top SHAP : " +
          " | ".join(f"{k}={v:+.4f}" for k, v in top_shap))


# ════════════════════════════════════════════════════════════
# SECTION C — AGGREGATION (after Week 4)
# ════════════════════════════════════════════════════════════

def aggregate_records(records: list) -> dict:
    """
    Take 4 weekly records and build the full data dict
    expected by generate_report().
    """
    assert len(records) == 4, "Need exactly 4 weekly records."

    def mean_dicts(list_of_dicts):
        keys = [k for k, v in list_of_dicts[0].items() if v is not None]
        return {k: round(float(np.mean([d[k] for d in list_of_dicts
                                        if d.get(k) is not None])), 4)
                for k in keys}

    # ── Weekly summary rows (Section 3) ──────────────────────
    def _lf(r, key):
        return r["lifestyle_week"].get(key)
    def _cog(r, key):
        return r["cognitive_week"].get(key)

    weekly = [
        {
            "week":             r["week"],
            "pa_score":         r["pa_score"],
            "na_score":         r["na_score"],
            "mood_label":       r["mood_label"],
            "confidence_tier":  r["debate"]["confidence_tier"],
            "debate_outcome":   r["debate"]["debate_outcome"],
            "sleep":            _lf(r, "Sleep Quality"),
            "stress":           _lf(r, "Stress Level"),
            "focus":            _lf(r, "Focus Ability"),
            "fatigue":          _lf(r, "Physical Fatigue"),
            "rumination":       _cog(r, "Rumination"),
            "social":           _cog(r, "Social Connection"),
        }
        for r in records
    ]

    # ── Overall mood summary (Section 2) ─────────────────────
    pa_scores  = [r["pa_score"] for r in records]
    na_scores  = [r["na_score"] for r in records]
    balances   = [p - n for p, n in zip(pa_scores, na_scores)]
    avg_pa     = round(float(np.mean(pa_scores)), 4)
    avg_na     = round(float(np.mean(na_scores)), 4)
    bal_delta  = balances[-1] - balances[0]

    if bal_delta > 0.05:
        trend = "Improving"
    elif bal_delta < -0.05:
        trend = "Declining"
    else:
        trend = "Stable"

    # ── Averaged dimensions (Section 4) ──────────────────────
    pa_dims = mean_dicts([r["pa_dims_week"] for r in records])
    na_dims = mean_dicts([r["na_dims_week"] for r in records])

    # ── Cognitive (Section 5) ─────────────────────────────────
    cognitive = mean_dicts([r["cognitive_week"] for r in records])

    # ── Lifestyle (Section 6) ─────────────────────────────────
    lifestyle = mean_dicts([r["lifestyle_week"] for r in records])

    # ── Plutchik averaged (Section 7) ─────────────────────────
    plutchik = mean_dicts([r["plutchik_week"] for r in records])

    # ── Stability metrics (Section 8) ─────────────────────────
    variability_index = round(float(np.std(balances)), 4)
    labels = [r["mood_label"] for r in records]
    # Score: Stable=1.0, Moderate=0.5, Unstable=0.0 — average across weeks
    label_score_map = {"Stable": 1.0, "Moderate": 0.5, "Unstable": 0.0}
    stability_score = round(sum(label_score_map.get(l, 0.5) for l in labels) / len(labels), 2)
    label_dist = " | ".join(
        f"{lbl} × {labels.count(lbl)}"
        for lbl in ["Stable", "Moderate", "Unstable"]
        if labels.count(lbl) > 0
    )

    # ── SHAP averaged (Section 9) ─────────────────────────────
    shap_avg = {}
    for feat in MODEL_FEATURES:
        shap_avg[feat] = round(
            float(np.mean([abs(r["shap"][feat]) for r in records])), 6
        )

    # ── Model debate summary (new — Section 9 addition) ───────
    debate_summary = []
    for r in records:
        d = r["debate"]
        debate_summary.append({
            "week":            r["week"],
            "xgb_label":       d["xgb_label"],
            "xgb_confidence":  d["xgb_confidence"],
            "rf_label":        d["rf_label"],
            "rf_confidence":   d["rf_confidence"],
            "final_label":     d["final_label"],
            "final_confidence":d["final_confidence"],
            "confidence_tier": d["confidence_tier"],
            "debate_outcome":  d["debate_outcome"],
            "agreed":          d["agreed"],
            "class_probs":     d["class_probs"],
        })

    # ── Auto-generated interpretations ───────────────────────
    pa_interp  = _gen_pa_interpretation(pa_dims, pa_scores)
    na_interp  = _gen_na_interpretation(na_dims, na_scores)
    cog_interp = _gen_cognitive_interpretation(cognitive)
    lf_interp  = _gen_lifestyle_interpretation(lifestyle, na_scores)
    stab_interp= _gen_stability_interpretation(
                     variability_index, labels, balances)
    shap_interp= _gen_shap_interpretation(shap_avg, labels, records)
    patterns   = _gen_observed_patterns(
                     lifestyle, cognitive, na_scores, pa_scores,
                     na_dims, pa_dims, balances, labels, records)
    summary    = _gen_final_summary(
                     labels, pa_scores, na_scores, shap_avg,
                     lifestyle, cognitive, trend,
                     variability_index, stability_score, label_dist,
                     na_dims, pa_dims, balances, records)

    # ── Determine model description for Section 1 ─────────────
    agreements = sum(1 for r in records if r["debate"]["agreed"])
    model_desc = (
        f"XGBoost (primary) + Random Forest (debate) — "
        f"{agreements}/4 weeks unanimous"
    )

    # ── Per-week breakdowns (for multi-week tables in report) ────
    pa_dims_by_week    = [r["pa_dims_week"]    for r in records]
    na_dims_by_week    = [r["na_dims_week"]     for r in records]
    cognitive_by_week  = [r["cognitive_week"]   for r in records]
    lifestyle_by_week  = [r["lifestyle_week"]   for r in records]

    return {
        "user": {
            "user_id":       "ML-USER",
            "period_start":  "Week 1",
            "period_end":    "Week 4",
            "total_entries": 4,
            "model":         model_desc,
        },
        "mood_summary": {
            "avg_pa": avg_pa,
            "avg_na": avg_na,
            "trend":  trend,
        },
        "weekly":              weekly,
        "debate_summary":      debate_summary,
        "pa_dimensions":       pa_dims,
        "pa_dimensions_weekly":pa_dims_by_week,
        "pa_interpretation":   pa_interp,
        "na_dimensions":       na_dims,
        "na_dimensions_weekly":na_dims_by_week,
        "na_interpretation":   na_interp,
        "cognitive_indicators":        cognitive,
        "cognitive_indicators_weekly": cognitive_by_week,
        "cognitive_interpretation":    cog_interp,
        "lifestyle_factors":           lifestyle,
        "lifestyle_factors_weekly":    lifestyle_by_week,
        "lifestyle_interpretation":    lf_interp,
        "plutchik":            plutchik,
        "stability": {
            "variability_index":       variability_index,
            "stability_score":         stability_score,
            "label_distribution":      label_dist,
            "stability_interpretation":stab_interp,
        },
        "shap_importance":     shap_avg,
        "shap_interpretation": shap_interp,
        "observed_patterns":   patterns,
        "final_summary":       summary,
    }


# ── Interpretation generators ────────────────────────────────

def _gen_pa_interpretation(pa_dims, pa_scores):
    delta = pa_scores[-1] - pa_scores[0]
    top2  = sorted(pa_dims, key=pa_dims.get, reverse=True)[:2]
    if delta > 0.05:
        trend = "Energy and motivation increased gradually across the reporting period."
    elif delta < -0.05:
        trend = "Positive affect showed a decreasing trend across the period."
    else:
        trend = "Positive affect remained consistent across the period."
    return f"{trend} Highest-scoring dimensions: {top2[0]} and {top2[1]}."


def _gen_na_interpretation(na_dims, na_scores):
    delta = na_scores[-1] - na_scores[0]
    top2  = sorted(na_dims, key=na_dims.get, reverse=True)[:2]
    if delta < -0.05:
        week_drop = next((i+2 for i in range(len(na_scores)-1)
                          if na_scores[i+1] < na_scores[i]), 2)
        trend = f"Anxiety-related indicators decreased after Week {week_drop}."
    elif delta > 0.05:
        trend = "Negative affect indicators showed an increasing trend."
    else:
        trend = "Negative affect remained consistent across the period."
    return (f"{trend} Dominant negative drivers: "
            f"{top2[0]} and {top2[1]}.")


def _gen_cognitive_interpretation(cog):
    parts = []
    if cog.get("Rumination") is not None:
        parts.append(
            "Rumination was elevated throughout the period."
            if cog["Rumination"] > 0.6
            else "Rumination remained at moderate levels."
        )
    if cog.get("Social Connection") is not None and cog["Social Connection"] > 0.65:
        parts.append("Social connection remained consistently high.")
    if cog.get("Emotional Clarity") is not None and cog["Emotional Clarity"] > 0.5:
        parts.append("Emotional clarity was above the midpoint on average.")
    return " ".join(parts) if parts else "Cognitive indicators were within normal range."


def _gen_lifestyle_interpretation(lf, na_scores):
    parts = []
    stress  = lf.get("Stress Level")
    sleep   = lf.get("Sleep Quality")
    fatigue = lf.get("Physical Fatigue")
    focus   = lf.get("Focus Ability")

    # Sleep floor — scores at or near 0.0 are a critical concern, not just "low"
    if sleep is not None:
        if sleep <= 0.05:
            parts.append(
                "Sleep quality was at the absolute floor (0.0) across the entire period — "
                "the lowest possible score. This is a critical concern: severely disrupted "
                "sleep is one of the strongest known drivers of emotional dysregulation, "
                "elevated negative affect, and reduced cognitive performance. "
                "Addressing sleep should be the highest priority."
            )
        elif sleep < 0.5:
            parts.append(
                f"Sleep quality was below average (avg {round(sleep*100)}%), "
                "which likely contributed to elevated negative affect across the period."
            )
        elif sleep >= 0.75:
            parts.append(
                f"Sleep quality was good (avg {round(sleep*100)}%), "
                "providing a strong foundation for emotional stability."
            )

    # Stress
    if stress is not None:
        if stress > 0.65:
            parts.append(
                f"Stress was persistently high (avg {round(stress*100)}%), "
                "which is a primary driver of negative affect and mood instability."
            )
        elif stress > sleep if sleep is not None else False:
            parts.append(
                "Stress level was the strongest lifestyle predictor "
                "of negative affect across the reporting period."
            )

    # Fatigue
    if fatigue is not None and fatigue > 0.6:
        parts.append(
            f"Physical fatigue was significant (avg {round(fatigue*100)}%), "
            "which reduces emotional resilience and amplifies the impact of stressors."
        )

    # Focus
    if focus is not None and focus < 0.4:
        parts.append(
            f"Focus ability was low (avg {round(focus*100)}%), "
            "often a sign of cognitive overload or emotional exhaustion."
        )

    return " ".join(parts) if parts else "Lifestyle indicators showed no strong directional pattern."


def _gen_stability_interpretation(vi, labels, balances):
    if vi < 0.30:
        base = "Low variability — mood remained stable across the period."
    elif vi < 0.60:
        base = "Moderate variability — some emotional fluctuation observed."
    else:
        base = ("High variability — significant mood fluctuation detected. "
                "Consider monitoring more closely.")
    if labels[0] == "Unstable" and labels[-1] == "Stable":
        base += (" Positive trajectory observed — mood stabilised "
                 "over the reporting period.")
    elif all(l == "Stable" for l in labels):
        base += " Consistently stable across all four weeks."
    return base


def _gen_shap_interpretation(shap_avg, labels, records):
    top3 = sorted(shap_avg, key=shap_avg.get, reverse=True)[:3]
    base = (f"{top3[0]}, {top3[1]}, and {top3[2]} were the "
            f"strongest predictors of mood classification.")
    unstable_weeks = [r for r in records if r["mood_label"] == "Unstable"]
    if unstable_weeks:
        w      = unstable_weeks[0]
        driver = max(w["shap"], key=lambda k: abs(w["shap"][k]))
        base  += (f" In Week {w['week']}, {driver} was the primary "
                  f"driver of the Unstable classification.")
    return base


def _gen_observed_patterns(lf, cog, na_scores, pa_scores,
                           na_dims, pa_dims, balances, labels, records):
    """
    Generates rich, plain-English observed patterns from all available weekly data.
    Each pattern is a self-contained sentence a non-technical user can understand.
    """
    patterns = []

    def pct(v): return round(v * 100)  # normalised 0-1 → percentage for readability

    stress  = lf.get("Stress Level")
    sleep   = lf.get("Sleep Quality")
    fatigue = lf.get("Physical Fatigue")
    focus   = lf.get("Focus Ability")
    rum     = cog.get("Rumination")
    ec      = cog.get("Emotional Clarity")
    sc      = cog.get("Social Connection")
    safety  = cog.get("Psychological Safety")

    # ── Overall mood trajectory ────────────────────────────────────────────────
    best_week  = balances.index(max(balances)) + 1
    worst_week = balances.index(min(balances)) + 1
    pa_change  = pa_scores[-1] - pa_scores[0]
    na_change  = na_scores[-1] - na_scores[0]

    if pa_change > 0.05:
        patterns.append(
            f"Your energy and positive emotions (Positive Affect) grew steadily "
            f"across the four weeks — from {pct(pa_scores[0])}% in Week 1 to "
            f"{pct(pa_scores[-1])}% in Week 4. This is a meaningful improvement "
            f"and suggests your emotional reserves were building over time."
        )
    elif pa_change < -0.05:
        patterns.append(
            f"Your positive emotions (Positive Affect) declined across the period "
            f"— from {pct(pa_scores[0])}% in Week 1 down to {pct(pa_scores[-1])}% "
            f"in Week 4. This downward trend may reflect accumulating stress or fatigue."
        )
    else:
        patterns.append(
            f"Your positive emotions remained broadly consistent across the four weeks "
            f"(ranging from {pct(min(pa_scores))}% to {pct(max(pa_scores))}%), "
            f"suggesting a relatively steady emotional baseline."
        )

    if na_change < -0.05:
        patterns.append(
            f"Feelings of distress, anxiety, and irritability (Negative Affect) eased "
            f"as the weeks progressed — from {pct(na_scores[0])}% in Week 1 to "
            f"{pct(na_scores[-1])}% in Week 4. Reducing negative affect, even gradually, "
            f"has a meaningful positive effect on overall mood stability."
        )
    elif na_change > 0.05:
        patterns.append(
            f"Negative emotions such as stress, irritability, and anxiety intensified "
            f"over the period — from {pct(na_scores[0])}% in Week 1 to "
            f"{pct(na_scores[-1])}% in Week 4. It is worth identifying what changed "
            f"in your circumstances during this time."
        )

    # ── Best and worst week ────────────────────────────────────────────────────
    if best_week != worst_week:
        patterns.append(
            f"Week {best_week} was your strongest week emotionally "
            f"(mood balance: {balances[best_week-1]:+.2f}), while Week {worst_week} "
            f"was the most challenging (mood balance: {balances[worst_week-1]:+.2f}). "
            f"Understanding what was different in Week {best_week} — whether it was "
            f"better sleep, lower stress, or more social time — could help you replicate "
            f"those conditions."
        )

    # ── Stability label progression ────────────────────────────────────────────
    stable_count   = labels.count("Stable")
    unstable_count = labels.count("Unstable")
    if stable_count >= 3:
        patterns.append(
            f"You were classified as emotionally Stable in {stable_count} out of 4 weeks. "
            f"Stability means the model found your emotional responses to be proportionate "
            f"and consistent — a positive indicator of resilience."
        )
    elif unstable_count >= 2:
        patterns.append(
            f"You were classified as Unstable in {unstable_count} out of 4 weeks. "
            f"Instability does not mean something is wrong — it signals that your emotional "
            f"state was more reactive than usual, which is often tied to specific stressors "
            f"or life events during that period."
        )
    if labels[0] == "Unstable" and labels[-1] == "Stable":
        patterns.append(
            f"There is a clear positive story here: you started Week 1 as Unstable and "
            f"ended Week 4 as Stable. That arc — from instability to stability — reflects "
            f"genuine emotional recovery and improved coping across the period."
        )

    # ── Stress ────────────────────────────────────────────────────────────────
    if stress is not None:
        if stress > 0.65:
            patterns.append(
                f"Stress was consistently high throughout the period (avg {pct(stress)}% "
                f"of maximum). Chronic stress at this level tends to suppress positive "
                f"emotions and amplify negative ones — it is likely one of the main drivers "
                f"of your lower mood weeks."
            )
        elif stress < 0.35:
            patterns.append(
                f"Your stress levels were relatively low on average ({pct(stress)}% of "
                f"maximum), which is a protective factor. Low stress creates space for "
                f"positive emotions to emerge and helps maintain mood stability."
            )

    # ── Sleep ─────────────────────────────────────────────────────────────────
    if sleep is not None:
        if sleep < 0.4:
            patterns.append(
                f"Sleep quality averaged {pct(sleep)}% — below the recommended threshold. "
                f"Poor sleep directly impairs emotional regulation: it makes negative "
                f"emotions feel more intense and makes it harder to experience positive ones. "
                f"Improving sleep is one of the highest-leverage actions for mood stability."
            )
        elif sleep > 0.65:
            patterns.append(
                f"Sleep quality was good on average ({pct(sleep)}%), which is a strong "
                f"foundation for emotional stability. Good sleep helps the brain process "
                f"emotional experiences and recover from stress."
            )

    # ── Rumination ────────────────────────────────────────────────────────────
    if rum is not None:
        if rum > 0.6:
            patterns.append(
                f"Rumination — the tendency to replay negative thoughts repeatedly — was "
                f"elevated at {pct(rum)}% on average. High rumination is one of the "
                f"strongest predictors of prolonged negative mood. Practices such as "
                f"mindfulness, journalling, or brief physical activity can help interrupt "
                f"ruminative thought cycles."
            )
        elif rum < 0.35:
            patterns.append(
                f"Rumination was low ({pct(rum)}% on average), meaning you were not "
                f"getting caught in repetitive negative thought loops. This is a healthy "
                f"sign and contributes to faster emotional recovery."
            )

    # ── Social connection ─────────────────────────────────────────────────────
    if sc is not None:
        if sc > 0.65:
            patterns.append(
                f"Social connection scored {pct(sc)}% on average — a meaningful protective "
                f"factor. People with strong social ties tend to recover from stress faster "
                f"and report higher baseline mood. This is one of your key emotional strengths."
            )
        elif sc < 0.4:
            patterns.append(
                f"Social connection averaged only {pct(sc)}%, which is relatively low. "
                f"Feelings of isolation or disconnection from others can significantly "
                f"amplify negative emotions. Even small increases in social interaction "
                f"tend to have an outsized positive effect on mood."
            )

    # ── Fatigue ───────────────────────────────────────────────────────────────
    if fatigue is not None and fatigue > 0.6:
        patterns.append(
            f"Physical fatigue was notable at {pct(fatigue)}% on average. Fatigue and "
            f"emotional well-being are closely linked — persistent tiredness reduces your "
            f"capacity to regulate emotions and can make challenges feel more overwhelming "
            f"than they actually are."
        )

    # ── Emotional clarity ─────────────────────────────────────────────────────
    if ec is not None:
        if ec < 0.4:
            patterns.append(
                f"Emotional clarity — your ability to identify and understand what you are "
                f"feeling — averaged {pct(ec)}%, which is on the lower side. When we "
                f"struggle to name our emotions, it becomes harder to address them. "
                f"Simple habits like end-of-day emotional check-ins can help build this skill."
            )

    # ── Top NA driver ─────────────────────────────────────────────────────────
    if na_dims:
        top_na = max(na_dims, key=na_dims.get)
        top_na_val = na_dims[top_na]
        if top_na_val > 0.5:
            patterns.append(
                f"Among all negative emotions measured, '{top_na}' was the most prominent "
                f"(avg {pct(top_na_val)}%). This specific emotion — not just general "
                f"negativity — was the primary contributor to your negative affect score "
                f"across the period and is worth paying particular attention to."
            )

    # ── Top PA driver ─────────────────────────────────────────────────────────
    if pa_dims:
        top_pa = max(pa_dims, key=pa_dims.get)
        top_pa_val = pa_dims[top_pa]
        if top_pa_val > 0.5:
            patterns.append(
                f"Your strongest positive emotion was '{top_pa}' (avg {pct(top_pa_val)}%). "
                f"Positive emotions like this are not just pleasant — they actively build "
                f"psychological resilience and help buffer against the effects of stress."
            )

    return patterns if patterns else ["No strong directional patterns were detected across the reporting period."]


def _gen_final_summary(labels, pa_scores, na_scores, shap_avg,
                       lf, cog, trend,
                       variability_index, stability_score, label_dist,
                       na_dims, pa_dims, balances, records):
    """
    Generates a structured, plain-English final summary in 6 paragraphs.
    Written so a non-technical user understands what happened, why, and what to do.
    """
    def pct(v): return round(v * 100)

    # ── Core descriptors ──────────────────────────────────────────────────────
    first_label = labels[0]
    last_label  = labels[-1]
    avg_pa      = sum(pa_scores) / len(pa_scores)
    avg_na      = sum(na_scores) / len(na_scores)
    pa_change   = pa_scores[-1] - pa_scores[0]
    na_change   = na_scores[-1] - na_scores[0]
    best_week   = balances.index(max(balances)) + 1
    worst_week  = balances.index(min(balances)) + 1
    balance_avg = avg_pa - avg_na

    stress  = lf.get("Stress Level") or 0
    sleep   = lf.get("Sleep Quality") or 0
    fatigue = lf.get("Physical Fatigue") or 0
    focus   = lf.get("Focus Ability") or 0
    rum     = cog.get("Rumination") or 0
    sc      = cog.get("Social Connection") or 0
    ec      = cog.get("Emotional Clarity") or 0

    # ── Trajectory language ────────────────────────────────────────────────────
    if first_label != last_label:
        traj_phrase = (
            f"progressed from {first_label} in Week 1 to {last_label} by Week 4 — "
            + ("a clear positive development" if labels[-1] == "Stable"
               else "a shift that warrants closer attention")
        )
    elif all(l == first_label for l in labels):
        traj_phrase = f"remained consistently {first_label} across all four weeks"
    else:
        traj_phrase = (
            f"fluctuated across the four-week period "
            f"(distribution: {label_dist})"
        )

    # ── PA/NA language ─────────────────────────────────────────────────────────
    pa_verb = ("grew" if pa_change > 0.05 else "held steady" if abs(pa_change) <= 0.05 else "declined")
    na_verb = ("eased" if na_change < -0.05 else "held steady" if abs(na_change) <= 0.05 else "intensified")
    pa_detail = f"from {pct(pa_scores[0])}% to {pct(pa_scores[-1])}%"
    na_detail = f"from {pct(na_scores[0])}% to {pct(na_scores[-1])}%"

    # ── SHAP top features ──────────────────────────────────────────────────────
    top3_shap = sorted(shap_avg, key=shap_avg.get, reverse=True)[:3]
    top_feat  = top3_shap[0]
    # Make feature names readable
    def readable(f):
        return f.replace("Q1_","").replace("Q2_","").replace("Q4_","")                 .replace("Q7_","").replace("Q9_","").replace("Q11_","")                 .replace("Q15_","").replace("Q18_","").replace("_"," ")

    top3_readable = [readable(f) for f in top3_shap]

    # ── Model debate summary ───────────────────────────────────────────────────
    agreements  = sum(1 for r in records if r["debate"]["agreed"])
    high_conf_w = [r["week"] for r in records if r["debate"]["confidence_tier"] == "HIGH"]
    low_conf_w  = [r["week"] for r in records if r["debate"]["confidence_tier"] == "LOW"]

    # ── Lifestyle dominant concern factor ────────────────────────────────────────
    # For sleep and focus, lower score = higher concern (inverted).
    # For stress and fatigue, higher score = higher concern (normal).
    # We compute a "concern score" so max() reliably finds the biggest problem.
    def _concern(key, val):
        if val is None: return 0.0
        if key in ("sleep", "focus"):
            return 1.0 - val   # 0.0 sleep → concern=1.0 (maximum concern)
        return val             # high stress/fatigue → high concern

    lf_concern_scores = {
        "stress":  _concern("stress",   stress  if stress  is not None else 0.0),
        "sleep":   _concern("sleep",    sleep   if sleep   is not None else 0.0),
        "fatigue": _concern("fatigue",  fatigue if fatigue is not None else 0.0),
        "focus":   _concern("focus",    focus   if focus   is not None else 0.0),
    }
    top_lf = max(lf_concern_scores, key=lf_concern_scores.get)
    lf_concern_map = {
        "stress":  ("stress",          "ongoing stress tends to narrow emotional range and amplify negativity"),
        "sleep":   ("sleep quality",   "poor sleep directly impairs your brain's ability to regulate emotions"),
        "fatigue": ("physical fatigue","fatigue reduces emotional resilience and makes challenges feel harder"),
        "focus":   ("focus ability",   "difficulty concentrating often co-occurs with emotional strain"),
    }
    lf_name, lf_explanation = lf_concern_map[top_lf]

    # ── Recommendation logic ───────────────────────────────────────────────────
    if trend == "Improving" and last_label == "Stable":
        recommendation = (
            f"The single most important thing right now is to protect what is working. "
            f"You improved across the period — identify the specific habits or circumstances "
            f"from Week {best_week} (your best week) and be intentional about maintaining them. "
            + (f"Keep managing your {lf_name}, as it was your most prominent lifestyle factor. " if top_lf else "")
            + "Small consistent actions tend to have a compounding positive effect on mood stability."
        )
    elif trend == "Declining" or last_label == "Unstable":
        recommendation = (
            f"The decline observed across the period signals that something needs to change. "
            f"Start with your most actionable lever: {lf_name} — {lf_explanation}. "
            + (f"Rumination was also elevated (avg {pct(rum)}%), which tends to sustain negative mood cycles. "
               f"Brief mindfulness or journalling practices can help interrupt this pattern. "
               if rum > 0.55 else "")
            + f"If you are experiencing persistent low mood, speaking with a mental health professional "
            f"is a practical and effective next step — not a sign of weakness."
        )
    else:
        # Even in "Stable" trend, flag critical lifestyle issues clearly
        sleep_critical = sleep is not None and sleep <= 0.05
        pa_declining   = pa_scores[-1] < pa_scores[0] - 0.05
        if sleep_critical or pa_declining:
            recommendation = (
                f"While the overall mood trend was broadly stable, there are important "
                f"signals that need attention. "
                + (f"Sleep quality was at the floor across the entire period — "
                   f"this is the single highest-leverage area to address. "
                   f"Even a modest improvement in sleep quality tends to have an "
                   f"immediate positive effect on emotional stability. " if sleep_critical else "")
                + (f"Positive affect declined across the period (from {pct(pa_scores[0])}% to "
                   f"{pct(pa_scores[-1])}%), signalling a gradual erosion of emotional resources "
                   f"that may accelerate if left unaddressed. " if pa_declining else "")
                + (f"Reducing rumination (currently avg {pct(rum)}%) through brief "
                   f"mindfulness or journalling practices would also help." if rum > 0.45 else "")
            )
        else:
            recommendation = (
                f"Your mood was broadly stable across the period, which is a solid foundation. "
                f"To move from stable to thriving, focus on your key growth areas: "
                + (f"{lf_name} ({lf_explanation})" if top_lf else "lifestyle balance")
                + (f", and consider reducing rumination (currently avg {pct(rum)}%) "
                   f"through structured reflection practices." if rum > 0.45 else ".")
            )

    # ── Build paragraphs ───────────────────────────────────────────────────────
    paragraphs = []

    # P1 — Overview
    paragraphs.append(
        "OVERVIEW\n"
        f"Over the four-week period, your emotional state {traj_phrase}. "
        f"On average, your positive emotions (energy, enthusiasm, motivation) were at "
        f"{pct(avg_pa)}% of their maximum, while negative emotions (stress, anxiety, "
        f"irritability) were at {pct(avg_na)}% — giving an overall mood balance of "
        f"{balance_avg:+.2f}. "
        + ("This positive balance means your emotional resources were, on average, "
           "outweighing your emotional burdens."
           if balance_avg > 0
           else "This negative balance indicates your emotional burdens were outweighing "
                "your emotional resources during this period.")
    )

    # P2 — Week-by-week trajectory
    paragraphs.append(
        "WHAT CHANGED ACROSS THE WEEKS\n"
        f"Your positive emotions {pa_verb} across the period ({pa_detail}), "
        f"while negative emotions {na_verb} ({na_detail}). "
        f"Your emotionally strongest week was Week {best_week} "
        f"(mood balance: {balances[best_week-1]:+.2f}), and your most challenging was "
        f"Week {worst_week} (mood balance: {balances[worst_week-1]:+.2f}). "
        + (f"The variability index was {variability_index:.2f} — "
           + ("low variability, meaning your mood was consistent week to week."
              if variability_index < 0.3 else
              "moderate variability, meaning there were noticeable ups and downs."
              if variability_index < 0.6 else
              "high variability, meaning your mood shifted significantly between weeks.")
          )
    )

    # P3 — What the model found
    paragraphs.append(
        "WHAT THE MODEL DETECTED\n"
        f"The AI model analysed your responses each week and classified your mood stability "
        f"as: {label_dist}. "
        f"The two models (XGBoost and Random Forest) agreed unanimously in {agreements} out "
        f"of 4 weeks"
        + (f", with high confidence in Week(s) {', '.join(map(str, high_conf_w))}." if high_conf_w else ".")
        + f" The features that most influenced these classifications were: "
        f"{top3_readable[0]}, {top3_readable[1]}, and {top3_readable[2]}. "
        f"In plain terms, how {top3_readable[0].lower()} you reported feeling each week "
        f"was the single strongest predictor of whether you would be classified as "
        f"Stable, Moderate, or Unstable."
    )

    # P4 — Lifestyle and cognitive context
    lifestyle_notes = []
    if stress > 0.55:
        lifestyle_notes.append(f"high stress ({pct(stress)}%)")
    if sleep < 0.45:
        lifestyle_notes.append(f"below-average sleep quality ({pct(sleep)}%)")
    if rum > 0.55:
        lifestyle_notes.append(f"elevated rumination ({pct(rum)}%)")
    if sc > 0.6:
        lifestyle_notes.append(f"strong social connection ({pct(sc)}%) — a protective factor")
    if fatigue > 0.6:
        lifestyle_notes.append(f"significant physical fatigue ({pct(fatigue)}%)")

    if lifestyle_notes:
        paragraphs.append(
            "LIFESTYLE AND MENTAL CONTEXT\n"
            f"Your lifestyle data highlights several factors that shaped your mood this period: "
            f"{'; '.join(lifestyle_notes)}. "
            + (f"Of these, {lf_name} was the most prominent ({lf_explanation}). " if top_lf else "")
            + (f"Your social connection score ({pct(sc)}%) stands out as a genuine strength — "
               f"people with strong social ties show greater emotional resilience and faster "
               f"recovery from stressful periods. " if sc > 0.6 else "")
            + (f"Emotional clarity averaged {pct(ec)}%, "
               + ("which is healthy and helps you address emotions constructively."
                  if ec > 0.5 else
                  "which is an area for growth — being able to name your feelings "
                  "is the first step to managing them.")
               if ec else "")
        )

    # P5 — Recommendation
    paragraphs.append(
        "WHAT TO DO NEXT\n"
        + recommendation
    )

    # P6 — Disclaimer
    paragraphs.append(
        "IMPORTANT NOTE\n"
        f"This report was generated by an automated mood stability model trained on "
        f"population-level data. It is designed for personal wellness awareness and "
        f"self-reflection only. The classifications and interpretations in this report "
        f"do not constitute a clinical diagnosis, psychological assessment, or medical "
        f"advice. If you are experiencing persistent low mood, significant anxiety, or "
        f"any mental health concerns, please consult a qualified mental health professional."
    )

    return paragraphs


# ════════════════════════════════════════════════════════════
# SECTION D — COLOUR PALETTE + STYLES (report layer)
# ════════════════════════════════════════════════════════════

NAVY   = colors.HexColor("#1A237E")
BLUE   = colors.HexColor("#1565C0")
LBLUE  = colors.HexColor("#E3F2FD")
ORANGE = colors.HexColor("#E65100")
GREEN  = colors.HexColor("#1B5E20")
LGREEN = colors.HexColor("#E8F5E9")
GREY   = colors.HexColor("#37474F")
LGREY  = colors.HexColor("#F5F5F5")
WHITE  = colors.white
RED    = colors.HexColor("#C62828")
LORANGE= colors.HexColor("#FFF3E0")

mBLUE   = "#1565C0"
mRED    = "#C62828"
mGREEN  = "#1B5E20"
mORANGE = "#E65100"
mNAVY   = "#1A237E"


def make_styles():
    S = {}
    S["title"]      = ParagraphStyle("RT",   fontName="Helvetica-Bold", fontSize=22,
                                     textColor=NAVY, alignment=TA_CENTER, spaceAfter=4)
    S["subtitle"]   = ParagraphStyle("SUB",  fontName="Helvetica", fontSize=11,
                                     textColor=GREY, alignment=TA_CENTER, spaceAfter=16)
    S["section"]    = ParagraphStyle("SEC",  fontName="Helvetica-Bold", fontSize=13,
                                     textColor=WHITE, backColor=NAVY, leftIndent=8,
                                     spaceBefore=14, spaceAfter=6, leading=20,
                                     borderPadding=(4,6,4,6))
    S["subsection"] = ParagraphStyle("SSEC", fontName="Helvetica-Bold", fontSize=10,
                                     textColor=NAVY, spaceBefore=8, spaceAfter=4)
    S["body"]       = ParagraphStyle("BODY", fontName="Helvetica", fontSize=9,
                                     textColor=GREY, spaceAfter=4, leading=14)
    S["interp"]     = ParagraphStyle("INT",  fontName="Helvetica-Oblique", fontSize=9,
                                     textColor=GREEN, spaceAfter=6, leading=13,
                                     leftIndent=12, borderPadding=(3,6,3,6),
                                     backColor=LGREEN)
    S["debate_win"] = ParagraphStyle("DW",   fontName="Helvetica-Bold", fontSize=9,
                                     textColor=GREEN, spaceAfter=3, leading=13)
    S["debate_dis"] = ParagraphStyle("DD",   fontName="Helvetica-Oblique", fontSize=8,
                                     textColor=ORANGE, spaceAfter=3, leading=12,
                                     leftIndent=8)
    S["metric"]     = ParagraphStyle("MET",  fontName="Helvetica-Bold", fontSize=11,
                                     textColor=BLUE, alignment=TA_CENTER)
    S["label"]      = ParagraphStyle("LBL",  fontName="Helvetica", fontSize=8,
                                     textColor=GREY, alignment=TA_CENTER)
    S["footer"]     = ParagraphStyle("FTR",  fontName="Helvetica", fontSize=7,
                                     textColor=GREY, alignment=TA_CENTER)
    return S


def fig_to_image(fig, width_cm=16):
    from PIL import Image as PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    pil   = PILImage.open(buf)
    w_px, h_px = pil.size
    buf.seek(0)
    img = Image(buf)
    img.drawWidth  = width_cm * cm
    img.drawHeight = width_cm * cm * (h_px / w_px)
    return img


def section_header(text, S):
    return Paragraph(text, S["section"])


def header_table_style():
    return TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LGREY, WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ])


def kv_table_style():
    return TableStyle([
        ("FONTNAME",      (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",      (1,0),(1,-1), "Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("TEXTCOLOR",     (0,0),(0,-1), NAVY),
        ("TEXTCOLOR",     (1,0),(1,-1), GREY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [LGREY, WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ])


def debate_table_style():
    """Special style for the model debate table."""
    return TableStyle([
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LGREY, WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ])


# ════════════════════════════════════════════════════════════
# SECTION E — CHART GENERATORS
# ════════════════════════════════════════════════════════════

def chart_pa_na_trend(weekly_data):
    weeks   = [f"Week {r['week']}" for r in weekly_data]
    pa      = [r["pa_score"] for r in weekly_data]
    na      = [r["na_score"] for r in weekly_data]
    balance = [p - n for p, n in zip(pa, na)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(weeks, pa,      "o-",  color=mBLUE,   lw=2.2, label="Positive Affect (PA)", ms=7)
    ax.plot(weeks, na,      "s-",  color=mRED,    lw=2.2, label="Negative Affect (NA)", ms=7)
    ax.plot(weeks, balance, "^--", color=mGREEN,  lw=1.6, label="Mood Balance (PA−NA)", ms=6, alpha=0.75)
    ax.axhline(0, color="#aaa", lw=0.8, ls="--")
    ax.fill_between(weeks, pa, na,
                    where=[p >= n for p, n in zip(pa, na)],
                    alpha=0.08, color=mBLUE, interpolate=True)
    ax.fill_between(weeks, pa, na,
                    where=[p < n for p, n in zip(pa, na)],
                    alpha=0.08, color=mRED, interpolate=True)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Weekly PA / NA Trend", fontsize=12, fontweight="bold", color=mNAVY)
    ax.legend(fontsize=9)
    ax.set_ylim([-0.2, 1.1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=16)


def chart_mood_stability(weekly_data):
    lc     = {"Stable": mGREEN, "Moderate": mORANGE, "Unstable": mRED}
    weeks  = [f"Week {r['week']}" for r in weekly_data]
    labels = [r["mood_label"] for r in weekly_data]
    vals   = [{"Stable": 3, "Moderate": 2, "Unstable": 1}[l] for l in labels]
    clrs   = [lc[l] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(weeks, vals, color=clrs, width=0.5, edgecolor="white", linewidth=1.2)
    for bar, lbl in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.08, lbl,
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Unstable", "Moderate", "Stable"], fontsize=9)
    ax.set_ylim([0, 3.8])
    ax.set_title("Weekly Mood Stability Label", fontsize=11, fontweight="bold", color=mNAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(color=mGREEN,  label="Stable"),
        mpatches.Patch(color=mORANGE, label="Moderate"),
        mpatches.Patch(color=mRED,    label="Unstable"),
    ], fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig_to_image(fig, width_cm=14)


def chart_debate_confidence(debate_summary):
    """Grouped bar chart — XGBoost vs RF confidence per week."""
    weeks = [f"Week {d['week']}" for d in debate_summary]
    xgb_c = [d["xgb_confidence"] for d in debate_summary]
    rf_c  = [d["rf_confidence"]  for d in debate_summary]
    x     = np.arange(len(weeks))
    w     = 0.35

    fig, ax = plt.subplots(figsize=(9, 3.8))
    b1 = ax.bar(x - w/2, xgb_c, w, label="XGBoost", color=mBLUE,   alpha=0.85)
    b2 = ax.bar(x + w/2, rf_c,  w, label="Random Forest", color=mORANGE, alpha=0.85)

    # Mark unanimous weeks with a star
    for i, d in enumerate(debate_summary):
        if d["agreed"]:
            ax.text(x[i], max(xgb_c[i], rf_c[i]) + 0.02, "★",
                    ha="center", fontsize=12, color=mGREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(weeks, fontsize=9)
    ax.set_ylabel("Confidence", fontsize=9)
    ax.set_ylim([0, 1.12])
    ax.axhline(0.80, color="grey", lw=0.8, ls="--", label="High confidence (0.80)")
    ax.axhline(0.60, color="#ccc", lw=0.8, ls=":")
    ax.set_title("Model Confidence per Week  (★ = Unanimous agreement)",
                 fontsize=11, fontweight="bold", color=mNAVY)
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=16)


def chart_probability_heatmap(debate_summary):
    """Heatmap — per-class probability for XGBoost, all 4 weeks."""
    class_names = ["Stable", "Moderate", "Unstable"]
    weeks       = [f"Week {d['week']}" for d in debate_summary]
    data        = np.array([
        [d["class_probs"][c]["xgb"] for c in class_names]
        for d in debate_summary
    ])  # shape (4, 3)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticks(range(4))
    ax.set_yticklabels(weeks, fontsize=9)
    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{data[i,j]:.2f}",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if data[i,j] > 0.55 else "#333")
    plt.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title("XGBoost Class Probability per Week",
                 fontsize=11, fontweight="bold", color=mNAVY)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=12)


def chart_radar(scores_dict, title, color):
    labels = list(scores_dict.keys())
    vals   = list(scores_dict.values())
    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    vals  += vals[:1];  angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, alpha=0.18, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim([0, 1])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.50","0.75","1.00"], size=7, color="grey")
    ax.set_title(title, size=11, fontweight="bold", color=mNAVY, pad=18)
    ax.grid(color="grey", alpha=0.2)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=8)


def chart_plutchik_wheel(emotion_scores):
    emotions  = ["Joy","Trust","Fear","Surprise",
                 "Sadness","Disgust","Anger","Anticipation"]
    em_colors = ["#FDD835","#8BC34A","#66BB6A","#29B6F6",
                 "#5C6BC0","#AB47BC","#EF5350","#FF7043"]
    scores    = [emotion_scores.get(e, 0) for e in emotions]
    N         = len(emotions)
    angles    = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    width     = 2 * np.pi / N

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for angle, score, color, label in zip(angles, scores, em_colors, emotions):
        ax.bar(angle, score, width=width*0.85, bottom=0.1,
               color=color, alpha=0.75, edgecolor="white", linewidth=1.2)
        ax.text(angle, score + 0.15, label,
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#333")
    ax.set_ylim([0, 1.2])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25","0.5","0.75","1.0"], size=7, color="grey")
    ax.set_xticks([])
    ax.set_title("Plutchik Emotion Wheel", size=11,
                 fontweight="bold", color=mNAVY, pad=22)
    ax.grid(color="grey", alpha=0.15)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=9)


def chart_shap_bar(shap_dict):
    features = list(shap_dict.keys())
    values   = list(shap_dict.values())
    pairs    = sorted(zip(values, features))
    values, features = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 4))
    clrs = [mRED if v > 0.05 else mBLUE for v in values]
    bars = ax.barh(features, values, color=clrs, edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.0005,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("Feature Importance (SHAP — averaged across 4 weeks)",
                 fontsize=11, fontweight="bold", color=mNAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=14)


# ════════════════════════════════════════════════════════════
# SECTION F — PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════

def generate_report(data: dict, output_path="moodlens_report.pdf"):
    S   = make_styles()
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="MoodLens Mood Analysis Report",
        author="MoodLens System"
    )
    story = []

    # ── COVER ─────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Mood Analysis Report", S["title"]))
    story.append(Paragraph("User Mood Monitoring System — MoodLens", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=12))

    # ── SECTION 1: USER INFORMATION ───────────────────────────
    story.append(section_header("1. User Information", S))
    u = data["user"]
    t = Table([
        ["User ID",               u.get("user_id","—")],
        ["Report Period",         f"{u.get('period_start','—')} to {u.get('period_end','—')}"],
        ["Total Entries",         str(u.get("total_entries","—"))],
        ["Model Used",            u.get("model","XGBoost + RF")],
        ["Questionnaire Version", "Unified Mood Scale (30 items)"],
    ], colWidths=[5*cm, 11*cm])
    t.setStyle(kv_table_style())
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── SECTION 2: OVERALL MOOD SUMMARY ───────────────────────
    story.append(section_header("2. Overall Mood Summary", S))
    m   = data["mood_summary"]
    pa  = m["avg_pa"]
    na  = m["avg_na"]
    bal = round(pa - na, 4)

    mt = Table([[
        Paragraph(f"<b>{pa:.3f}</b>",  S["metric"]),
        Paragraph(f"<b>{na:.3f}</b>",  S["metric"]),
        Paragraph(f"<b>{bal:+.3f}</b>",S["metric"]),
    ],[
        Paragraph("Avg Positive Affect (PA)", S["label"]),
        Paragraph("Avg Negative Affect (NA)", S["label"]),
        Paragraph("Mood Balance (PA − NA)",   S["label"]),
    ]], colWidths=[5.3*cm]*3)
    mt.setStyle(TableStyle([
        ("ALIGN",        (0,0),(-1,-1),"CENTER"),
        ("BACKGROUND",   (0,0),(-1,0), LBLUE),
        ("TOPPADDING",   (0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("BOX",          (0,0),(-1,-1),0.8, BLUE),
        ("INNERGRID",    (0,0),(-1,-1),0.4, BLUE),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.3*cm))

    interp = ("Positive balance — overall positive mood state." if bal > 0.05
              else "Negative balance — emotional strain detected." if bal < -0.05
              else "Balanced — stable emotional state.")
    story.append(Paragraph(
        f"Interpretation: {interp}  |  Overall Trend: {m.get('trend','—')}",
        S["interp"]
    ))

    # ── SECTION 3: WEEKLY MOOD TREND ──────────────────────────
    story.append(section_header("3. Weekly Mood Trend", S))

    # ── 3A: Core mood scores table ────────────────────────────
    story.append(Paragraph("Mood Scores", S["subsection"]))
    label_color_map = {"Stable": GREEN, "Moderate": ORANGE, "Unstable": RED}
    conf_color_map  = {"HIGH": GREEN, "MEDIUM": BLUE, "LOW": RED}

    mood_rows = [["Week", "PA Score", "NA Score", "Balance", "Label", "Confidence"]]
    for w in data["weekly"]:
        b = round(w["pa_score"] - w["na_score"], 4)
        mood_rows.append([
            f"Week {w['week']}",
            f"{w['pa_score']:.3f}",
            f"{w['na_score']:.3f}",
            f"{b:+.3f}",
            w["mood_label"],
            w.get("confidence_tier", "—"),
        ])
    wt = Table(mood_rows, colWidths=[2.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 3.2*cm, 2.2*cm])
    # Build style with per-row label colouring
    wt_style = [
        ("BACKGROUND",    (0,0),(-1,0), NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ALIGN",         (0,0),(-1,-1), "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [LGREY, WHITE]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]
    for i, w in enumerate(data["weekly"], start=1):
        lbl_c = label_color_map.get(w["mood_label"], GREY)
        cfd_c = conf_color_map.get(w.get("confidence_tier",""), GREY)
        wt_style.append(("TEXTCOLOR", (4,i),(4,i), lbl_c))
        wt_style.append(("FONTNAME",  (4,i),(4,i), "Helvetica-Bold"))
        wt_style.append(("TEXTCOLOR", (5,i),(5,i), cfd_c))
        wt_style.append(("FONTNAME",  (5,i),(5,i), "Helvetica-Bold"))
    wt.setStyle(TableStyle(wt_style))
    story.append(wt)
    story.append(Spacer(1, 0.3*cm))

    # ── 3B: Lifestyle per-week table ──────────────────────────
    lf_keys = [
        ("Sleep",    "sleep"),
        ("Stress",   "stress"),
        ("Focus",    "focus"),
        ("Fatigue",  "fatigue"),
        ("Rumination","rumination"),
        ("Social",   "social"),
    ]
    has_lf = any(w.get("sleep") is not None for w in data["weekly"])
    if has_lf:
        story.append(Paragraph("Weekly Lifestyle & Context Snapshot", S["subsection"]))
        story.append(Paragraph(
            "Scores are on the normalised 0–1 scale. "
            "Higher = more intense for Stress, Fatigue, Rumination; "
            "higher = better for Sleep, Focus, Social.",
            S["body"]
        ))
        story.append(Spacer(1, 0.1*cm))

        lf_rows = [["Factor"] + [f"Week {w['week']}" for w in data["weekly"]]]
        for label, key in lf_keys:
            row = [label]
            for w in data["weekly"]:
                v = w.get(key)
                row.append(f"{v:.2f}" if v is not None else "—")
            lf_rows.append(row)

        n_weeks = len(data["weekly"])
        lf_col_w = (16*cm - 3.5*cm) / n_weeks
        lf_t = Table(lf_rows, colWidths=[3.5*cm] + [lf_col_w]*n_weeks)
        lf_t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
            ("FONTNAME",      (0,1),(0,-1), "Helvetica-Bold"),
            ("TEXTCOLOR",     (0,1),(0,-1), NAVY),
            ("FONTSIZE",      (0,0),(-1,-1), 9),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("ALIGN",         (0,1),(0,-1),  "LEFT"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [LGREY, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#BDBDBD")),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,1),(0,-1),  6),
        ]))
        story.append(lf_t)
        story.append(Spacer(1, 0.3*cm))

    # ── 3C: Charts ────────────────────────────────────────────
    story.append(chart_pa_na_trend(data["weekly"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_mood_stability(data["weekly"]))

    # ── SECTION 3B: MODEL DEBATE RESULTS ──────────────────────
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Model Debate Results", S["subsection"]))
    story.append(Paragraph(
        "Each week, XGBoost and Random Forest independently predict the stability label. "
        "The debate engine selects the final label based on agreement, confidence gap, "
        "and tie-break rules. ★ indicates unanimous agreement between both models.",
        S["body"]
    ))

    debate_rows = [["Week", "XGBoost", "XGB Conf.", "RF", "RF Conf.",
                    "Final Label", "Confidence", "Tier", "Agreed"]]
    for d in data.get("debate_summary", []):
        agreed_str = "★ Yes" if d["agreed"] else "No"
        debate_rows.append([
            f"Week {d['week']}",
            d["xgb_label"],
            f"{d['xgb_confidence']:.1%}",
            d["rf_label"],
            f"{d['rf_confidence']:.1%}",
            d["final_label"],
            f"{d['final_confidence']:.1%}",
            d["confidence_tier"],
            agreed_str,
        ])
    dt = Table(debate_rows,
               colWidths=[1.8*cm,2.0*cm,1.8*cm,2.0*cm,1.8*cm,
                           2.2*cm,2.0*cm,1.8*cm,1.6*cm])
    dt.setStyle(debate_table_style())
    story.append(dt)
    story.append(Spacer(1, 0.3*cm))

    # Confidence chart
    if data.get("debate_summary"):
        story.append(chart_debate_confidence(data["debate_summary"]))
        story.append(Spacer(1, 0.3*cm))
        story.append(chart_probability_heatmap(data["debate_summary"]))

    # ── SECTION 4: EMOTIONAL DIMENSIONS ───────────────────────
    story.append(section_header("4. Emotional Dimension Analysis", S))

    def weekly_table(items_avg, items_weekly, label_col, col_w_label=4.5*cm):
        """Build a table with Item | W1 | W2 | W3 | W4 | Avg columns."""
        n = len(items_weekly)
        week_headers = [f"W{i+1}" for i in range(n)]
        header = [label_col] + week_headers + ["Avg"]
        rows = [header]
        for key, avg_val in items_avg.items():
            row = [key]
            for wk in items_weekly:
                v = wk.get(key)
                row.append(f"{v:.2f}" if v is not None else "—")
            row.append(f"{avg_val:.2f}")
            rows.append(row)
        n_data_cols = n + 1  # week cols + avg
        data_col_w  = (16*cm - col_w_label) / (n_data_cols)
        col_widths  = [col_w_label] + [data_col_w] * n_data_cols
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0), WHITE),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTNAME",      (0,1),(-1,-1),"Helvetica"),
            ("FONTNAME",      (-1,1),(-1,-1),"Helvetica-Bold"),   # Avg col bold
            ("BACKGROUND",    (-1,1),(-1,-1), LBLUE),              # Avg col tinted
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ALIGN",         (0,0),(-1,-1), "CENTER"),
            ("ALIGN",         (0,1),(0,-1),  "LEFT"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-2,-1), [LGREY, WHITE]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#BDBDBD")),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
            ("LEFTPADDING",   (0,1),(0,-1),  6),
        ]))
        return t

    story.append(Paragraph("Positive Affect Dimensions", S["subsection"]))
    pa_items  = data["pa_dimensions"]
    pa_weekly = data.get("pa_dimensions_weekly", [])
    if pa_weekly:
        story.append(weekly_table(pa_items, pa_weekly, "Item"))
    else:
        pa_rows = [["Item","Avg Score"]] + [[k,f"{v:.3f}"] for k,v in pa_items.items()]
        story.append(Table(pa_rows, colWidths=[8*cm,8*cm], style=header_table_style()))
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_radar(pa_items, "Positive Affect Profile (4-Week Avg)", mBLUE))
    story.append(Paragraph(data.get("pa_interpretation",""), S["interp"]))

    story.append(Paragraph("Negative Affect Dimensions", S["subsection"]))
    na_items  = data["na_dimensions"]
    na_weekly = data.get("na_dimensions_weekly", [])
    if na_weekly:
        story.append(weekly_table(na_items, na_weekly, "Item"))
    else:
        na_rows = [["Item","Avg Score"]] + [[k,f"{v:.3f}"] for k,v in na_items.items()]
        story.append(Table(na_rows, colWidths=[8*cm,8*cm], style=header_table_style()))
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_radar(na_items, "Negative Affect Profile (4-Week Avg)", mRED))
    story.append(Paragraph(data.get("na_interpretation",""), S["interp"]))

    # ── SECTION 5: COGNITIVE ──────────────────────────────────
    story.append(section_header("5. Cognitive and Context Indicators", S))
    cog         = data["cognitive_indicators"]
    cog_weekly  = data.get("cognitive_indicators_weekly", [])
    if cog_weekly:
        story.append(weekly_table(cog, cog_weekly, "Indicator", col_w_label=5*cm))
    else:
        cog_rows = [["Indicator","Average Score"]] + [[k,f"{v:.3f}"] for k,v in cog.items()]
        story.append(Table(cog_rows, colWidths=[9*cm,7*cm], style=header_table_style()))
    story.append(Paragraph(data.get("cognitive_interpretation",""), S["interp"]))

    # ── SECTION 6: LIFESTYLE ──────────────────────────────────
    story.append(section_header("6. Lifestyle and Wellness Factors", S))
    lf          = data["lifestyle_factors"]
    lf_weekly   = data.get("lifestyle_factors_weekly", [])
    if lf_weekly:
        story.append(weekly_table(lf, lf_weekly, "Factor", col_w_label=5*cm))
    else:
        lf_rows = [["Factor","Average Score"]] + [[k,f"{v:.3f}"] for k,v in lf.items()]
        story.append(Table(lf_rows, colWidths=[9*cm,7*cm], style=header_table_style()))
    story.append(Paragraph(data.get("lifestyle_interpretation",""), S["interp"]))

    # ── SECTION 7: EMOTIONAL PROFILE ──────────────────────────
    story.append(section_header("7. Emotional Profile Visualisation", S))
    story.append(Paragraph(
        "The Plutchik Wheel shows the relative strength of eight primary emotions "
        "averaged across the four-week reporting period. Petal length corresponds "
        "to mean intensity on the normalised [0,1] scale.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(chart_plutchik_wheel(data["plutchik"]))
    story.append(Paragraph(
        "The wheel highlights dominant emotions and overall emotional balance over time.",
        S["interp"]
    ))

    # ── SECTION 8: STABILITY ──────────────────────────────────
    story.append(section_header("8. Mood Stability Analysis", S))
    ms  = data["stability"]
    st  = Table([
        ["Mood Variability Index",    f"{ms.get('variability_index','—')}"],
        ["Emotional Stability Score", f"{ms.get('stability_score','—')}"],
        ["Label Distribution",        ms.get("label_distribution","—")],
    ], colWidths=[7*cm, 9*cm])
    st.setStyle(kv_table_style())
    story.append(st)
    story.append(Paragraph(ms.get("stability_interpretation",""), S["interp"]))

    # ── SECTION 9: MODEL INSIGHTS ─────────────────────────────
    story.append(section_header("9. Model Insights (SHAP Feature Importance)", S))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's "
        "contribution to the XGBoost prediction. Values shown are mean absolute "
        "SHAP importance averaged across all four weeks.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(chart_shap_bar(data["shap_importance"]))
    story.append(Paragraph(data.get("shap_interpretation",""), S["interp"]))

    # ── SECTION 10: PATTERNS ──────────────────────────────────
    story.append(section_header("10. Observed Patterns", S))
    story.append(Paragraph(
        "The following patterns were identified from your four weeks of data. "
        "Each point is drawn directly from the numbers — not generic advice.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    for p in data.get("observed_patterns",[]):
        story.append(Paragraph(f"• {p}", S["body"]))
        story.append(Spacer(1, 0.1*cm))
    story.append(Spacer(1, 0.2*cm))

    # ── SECTION 11: SUMMARY ───────────────────────────────────
    story.append(section_header("11. Final Summary", S))
    final = data.get("final_summary", "No summary provided.")
    if isinstance(final, list):
        for para in final:
            # If paragraph starts with an ALL-CAPS heading line, split and style it
            if "\n" in para:
                heading, body_text = para.split("\n", 1)
                story.append(Paragraph(heading, S["subsection"]))
                story.append(Paragraph(body_text, S["body"]))
            elif para.isupper() or (len(para) < 40 and para == para.upper()):
                story.append(Paragraph(para, S["subsection"]))
            else:
                # Check if first line is a heading (ends before first sentence)
                lines = para.split("\n", 1)
                if len(lines) == 2 and lines[0] == lines[0].upper() and len(lines[0]) < 50:
                    story.append(Paragraph(lines[0], S["subsection"]))
                    story.append(Paragraph(lines[1], S["body"]))
                else:
                    story.append(Paragraph(para, S["body"]))
            story.append(Spacer(1, 0.3*cm))
    else:
        story.append(Paragraph(final, S["body"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=NAVY, spaceAfter=6))
    story.append(Paragraph(
        "Generated by MoodLens · For personal wellness monitoring only · "
        "Not a clinical diagnosis · Consult a qualified professional for medical advice.",
        S["footer"]
    ))

    doc.build(story)
    print(f"\nReport saved: {output_path}")


# ════════════════════════════════════════════════════════════
# SECTION G — FULL PIPELINE ENTRY POINT
# ════════════════════════════════════════════════════════════

def generate_report_from_records(records: list, output_path="moodlens_report.pdf"):
    """
    Aggregates 4 weekly records and generates the PDF.
    This is the function to call from Flask after Week 4.
    """
    assert len(records) == 4, "Need exactly 4 records."
    data = aggregate_records(records)
    generate_report(data, output_path)
    return data


# ════════════════════════════════════════════════════════════
# DEMO — runs with dummy models when called directly
# ════════════════════════════════════════════════════════════

def _build_demo_model():
    """
    Demo substitute using sklearn GradientBoosting (no network install needed).
    In production replace with:
        xgb_model = joblib.load("mood_stability_xgb_modelB.pkl")
        rf_model  = joblib.load("mood_stability_rf_modelB.pkl")
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=600, n_features=9,
                                n_classes=3, n_informative=6,
                                random_state=42)
    xgb_m = GradientBoostingClassifier(n_estimators=80, random_state=42)
    rf_m  = RandomForestClassifier(n_estimators=80, random_state=42)
    xgb_m.fit(X, y)
    rf_m.fit(X, y)
    return xgb_m, rf_m


# Four weeks of sample 30-item responses (Likert 1–5)
SAMPLE_RESPONSES = [
    # Week 1 — higher stress / lower PA
    {
        "Q1_Interested":1.0,"Q2_Distressed":4.0,"Q3_Excited":1.0,
        "Q4_Upset":4.0,"Q5_Strong":2.0,"Q6_Guilty":3.0,
        "Q7_Scared":3.0,"Q8_Hostile":2.0,"Q9_Enthusiastic":1.0,
        "Q10_Proud":2.0,"Q11_Irritable":4.0,"Q12_Alert":2.0,
        "Q13_Ashamed":2.0,"Q14_Inspired":1.0,"Q15_Nervous":4.0,
        "Q16_Determined":2.0,"Q17_Attentive":2.0,"Q18_Jittery":3.0,
        "Q19_Active":2.0,"Q20_Afraid":3.0,"Q21_Emotional_Clarity":2.0,
        "Sleep_Quality":2.0,"Daily_Stress":4.0,"Social_Connection":2.0,
        "Focus_Ability":2.0,"Physical_Fatigue":4.0,
        "Rumination":4.0,"Somatic_Awareness":3.0,"Psychological_Safety":2.0,
    },
    # Week 2 — slightly better
    {
        "Q1_Interested":2.0,"Q2_Distressed":3.0,"Q3_Excited":2.0,
        "Q4_Upset":3.0,"Q5_Strong":2.0,"Q6_Guilty":2.0,
        "Q7_Scared":2.0,"Q8_Hostile":2.0,"Q9_Enthusiastic":2.0,
        "Q10_Proud":2.0,"Q11_Irritable":3.0,"Q12_Alert":3.0,
        "Q13_Ashamed":2.0,"Q14_Inspired":2.0,"Q15_Nervous":3.0,
        "Q16_Determined":3.0,"Q17_Attentive":3.0,"Q18_Jittery":2.0,
        "Q19_Active":2.0,"Q20_Afraid":2.0,"Q21_Emotional_Clarity":3.0,
        "Sleep_Quality":3.0,"Daily_Stress":3.0,"Social_Connection":3.0,
        "Focus_Ability":3.0,"Physical_Fatigue":3.0,
        "Rumination":3.0,"Somatic_Awareness":2.0,"Psychological_Safety":3.0,
    },
    # Week 3 — improving
    {
        "Q1_Interested":3.0,"Q2_Distressed":2.0,"Q3_Excited":3.0,
        "Q4_Upset":2.0,"Q5_Strong":3.0,"Q6_Guilty":2.0,
        "Q7_Scared":2.0,"Q8_Hostile":1.0,"Q9_Enthusiastic":3.0,
        "Q10_Proud":3.0,"Q11_Irritable":2.0,"Q12_Alert":3.0,
        "Q13_Ashamed":1.0,"Q14_Inspired":3.0,"Q15_Nervous":2.0,
        "Q16_Determined":3.0,"Q17_Attentive":3.0,"Q18_Jittery":2.0,
        "Q19_Active":3.0,"Q20_Afraid":2.0,"Q21_Emotional_Clarity":3.0,
        "Sleep_Quality":3.0,"Daily_Stress":2.0,"Social_Connection":4.0,
        "Focus_Ability":3.0,"Physical_Fatigue":2.0,
        "Rumination":2.0,"Somatic_Awareness":2.0,"Psychological_Safety":4.0,
    },
    # Week 4 — stable / positive
    {
        "Q1_Interested":4.0,"Q2_Distressed":1.0,"Q3_Excited":4.0,
        "Q4_Upset":1.0,"Q5_Strong":4.0,"Q6_Guilty":1.0,
        "Q7_Scared":1.0,"Q8_Hostile":1.0,"Q9_Enthusiastic":4.0,
        "Q10_Proud":4.0,"Q11_Irritable":1.0,"Q12_Alert":4.0,
        "Q13_Ashamed":1.0,"Q14_Inspired":4.0,"Q15_Nervous":1.0,
        "Q16_Determined":4.0,"Q17_Attentive":4.0,"Q18_Jittery":1.0,
        "Q19_Active":4.0,"Q20_Afraid":1.0,"Q21_Emotional_Clarity":4.0,
        "Sleep_Quality":4.0,"Daily_Stress":1.0,"Social_Connection":5.0,
        "Focus_Ability":4.0,"Physical_Fatigue":1.0,
        "Rumination":1.0,"Somatic_Awareness":2.0,"Psychological_Safety":5.0,
    },
]


if __name__ == "__main__":
    print("Building demo models...")
    xgb_model, rf_model = _build_demo_model()

    print("\nRunning weekly inference for 4 weeks...")
    records = []
    for week_num, responses in enumerate(SAMPLE_RESPONSES, start=1):
        record = run_weekly_inference(responses, week_num, xgb_model, rf_model)
        records.append(record)

    print("\nGenerating PDF report...")
    generate_report_from_records(
        records,
        output_path="moodlens_report_v2.pdf"
    )