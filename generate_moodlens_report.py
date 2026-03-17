"""
generate_moodlens_report.py
============================
MoodLens Longitudinal Wellness Report — PDF Generator
Covers all 11 sections from the Report.docx specification:
  1.  User Information
  2.  Overall Mood Summary
  3.  Weekly Mood Trend (table + line chart)
  4.  Emotional Dimension Analysis (PA + NA)
  5.  Cognitive and Context Indicators
  6.  Lifestyle and Wellness Factors
  7.  Emotional Profile Visualisation (Plutchik Wheel + charts)
  8.  Mood Stability Analysis
  9.  Model Insights (SHAP feature importance)
  10. Observed Patterns
  11. Final Summary

Usage:
    python generate_moodlens_report.py

The function `generate_report(data)` is the main entry point.
Pass a dict — see SAMPLE_DATA at the bottom for the full schema.
Output: moodlens_report.pdf
"""

# ── Standard library ────────────────────────────────────────
import io
import math

# ── Third-party ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image, PageBreak, KeepTogether
)
from reportlab.platypus.flowables import Flowable

# ════════════════════════════════════════════════════════════
# COLOUR PALETTE
# ════════════════════════════════════════════════════════════
NAVY    = colors.HexColor("#1A237E")
BLUE    = colors.HexColor("#1565C0")
LBLUE   = colors.HexColor("#E3F2FD")
ORANGE  = colors.HexColor("#E65100")
GREEN   = colors.HexColor("#1B5E20")
LGREEN  = colors.HexColor("#E8F5E9")
GREY    = colors.HexColor("#37474F")
LGREY   = colors.HexColor("#F5F5F5")
WHITE   = colors.white
RED     = colors.HexColor("#C62828")

# matplotlib hex equivalents
mBLUE   = "#1565C0"
mRED    = "#C62828"
mGREEN  = "#1B5E20"
mORANGE = "#E65100"
mNAVY   = "#1A237E"
mLBLUE  = "#E3F2FD"

# ════════════════════════════════════════════════════════════
# STYLES
# ════════════════════════════════════════════════════════════
base_styles = getSampleStyleSheet()

def make_styles():
    S = {}
    S["title"] = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold", fontSize=22,
        textColor=NAVY, alignment=TA_CENTER, spaceAfter=4
    )
    S["subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica", fontSize=11,
        textColor=GREY, alignment=TA_CENTER, spaceAfter=16
    )
    S["section"] = ParagraphStyle(
        "Section",
        fontName="Helvetica-Bold", fontSize=13,
        textColor=WHITE, backColor=NAVY,
        leftIndent=8, rightIndent=8,
        spaceBefore=14, spaceAfter=6,
        leading=20, borderPadding=(4, 6, 4, 6)
    )
    S["subsection"] = ParagraphStyle(
        "Subsection",
        fontName="Helvetica-Bold", fontSize=10,
        textColor=NAVY, spaceBefore=8, spaceAfter=4
    )
    S["body"] = ParagraphStyle(
        "Body",
        fontName="Helvetica", fontSize=9,
        textColor=GREY, spaceAfter=4, leading=14
    )
    S["interp"] = ParagraphStyle(
        "Interp",
        fontName="Helvetica-Oblique", fontSize=9,
        textColor=GREEN, spaceAfter=6, leading=13,
        leftIndent=12, borderPadding=(3, 6, 3, 6),
        backColor=LGREEN
    )
    S["metric"] = ParagraphStyle(
        "Metric",
        fontName="Helvetica-Bold", fontSize=11,
        textColor=BLUE, alignment=TA_CENTER
    )
    S["label"] = ParagraphStyle(
        "Label",
        fontName="Helvetica", fontSize=8,
        textColor=GREY, alignment=TA_CENTER
    )
    S["footer"] = ParagraphStyle(
        "Footer",
        fontName="Helvetica", fontSize=7,
        textColor=GREY, alignment=TA_CENTER
    )
    return S

# ════════════════════════════════════════════════════════════
# HELPER: matplotlib figure → ReportLab Image in memory
# ════════════════════════════════════════════════════════════
def fig_to_image(fig, width_cm=16):
    from PIL import Image as PILImage
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    plt.close(fig)
    pil = PILImage.open(buf)
    w_px, h_px = pil.size
    buf.seek(0)
    img = Image(buf)
    img.drawWidth  = width_cm * cm
    img.drawHeight = width_cm * cm * (h_px / w_px)
    return img

# ════════════════════════════════════════════════════════════
# HELPER: section header
# ════════════════════════════════════════════════════════════
def section_header(text, S):
    return Paragraph(text, S["section"])

# ════════════════════════════════════════════════════════════
# CHART GENERATORS
# ════════════════════════════════════════════════════════════

def chart_pa_na_trend(weekly_data):
    """Line chart — PA and NA scores across 4 weeks."""
    weeks  = [f"Week {r['week']}" for r in weekly_data]
    pa     = [r["pa_score"] for r in weekly_data]
    na     = [r["na_score"] for r in weekly_data]
    balance= [p - n for p, n in zip(pa, na)]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(weeks, pa,      "o-", color=mBLUE,   lw=2.2, label="Positive Affect (PA)", ms=7)
    ax.plot(weeks, na,      "s-", color=mRED,    lw=2.2, label="Negative Affect (NA)", ms=7)
    ax.plot(weeks, balance, "^--",color=mGREEN,  lw=1.6, label="Mood Balance (PA−NA)", ms=6, alpha=0.75)
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
    ax.set_ylim([-1, 5.5])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=16)


def chart_mood_stability(weekly_data):
    """Bar chart showing mood stability label per week."""
    label_color = {"Stable": mGREEN, "Moderate": mORANGE, "Unstable": mRED}
    weeks  = [f"Week {r['week']}" for r in weekly_data]
    labels = [r["mood_label"] for r in weekly_data]
    vals   = [{"Stable": 3, "Moderate": 2, "Unstable": 1}[l] for l in labels]
    clrs   = [label_color[l] for l in labels]

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
    legend_patches = [
        mpatches.Patch(color=mGREEN,  label="Stable"),
        mpatches.Patch(color=mORANGE, label="Moderate"),
        mpatches.Patch(color=mRED,    label="Unstable"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper right")
    plt.tight_layout()
    return fig_to_image(fig, width_cm=14)


def chart_radar(scores_dict, title, color):
    """Radar / spider chart for PA or NA dimensions."""
    labels = list(scores_dict.keys())
    vals   = list(scores_dict.values())
    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    vals  += vals[:1]
    angles+= angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, color=color, lw=2)
    ax.fill(angles, vals, alpha=0.18, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim([0, 4])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1","2","3","4"], size=7, color="grey")
    ax.set_title(title, size=11, fontweight="bold", color=mNAVY, pad=18)
    ax.grid(color="grey", alpha=0.2)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=8)


def chart_plutchik_wheel(emotion_scores):
    """
    Simplified Plutchik polar petal chart.
    emotion_scores: dict of 8 primary emotions → score (0–4)
    """
    emotions = [
        "Joy","Trust","Fear","Surprise",
        "Sadness","Disgust","Anger","Anticipation"
    ]
    em_colors = [
        "#FDD835","#8BC34A","#66BB6A","#29B6F6",
        "#5C6BC0","#AB47BC","#EF5350","#FF7043"
    ]
    scores = [emotion_scores.get(e, 0) for e in emotions]
    N      = len(emotions)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    width = 2 * np.pi / N

    for i, (angle, score, color, label) in enumerate(
        zip(angles, scores, em_colors, emotions)
    ):
        bar = ax.bar(
            angle, score, width=width * 0.85,
            bottom=0.3, color=color, alpha=0.75,
            edgecolor="white", linewidth=1.2
        )
        ax.text(
            angle, score + 0.55, label,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="#333"
        )

    ax.set_ylim([0, 4.8])
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["1","2","3","4"], size=7, color="grey")
    ax.set_xticks([])
    ax.set_title("Plutchik Emotion Wheel", size=11,
                 fontweight="bold", color=mNAVY, pad=22)
    ax.grid(color="grey", alpha=0.15)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=9)


def chart_shap_bar(shap_dict):
    """Horizontal bar chart of mean absolute SHAP values."""
    features = list(shap_dict.keys())
    values   = list(shap_dict.values())
    sorted_pairs = sorted(zip(values, features))
    values, features = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(features, values,
                   color=[mRED if v > 0.05 else mBLUE for v in values],
                   edgecolor="white", height=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title("Feature Importance (SHAP)", fontsize=11,
                 fontweight="bold", color=mNAVY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    return fig_to_image(fig, width_cm=14)


# ════════════════════════════════════════════════════════════
# TABLE STYLES
# ════════════════════════════════════════════════════════════
def header_table_style():
    return TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), NAVY),
        ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,0), 9),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,1), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[LGREY, WHITE]),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ])

def kv_table_style():
    return TableStyle([
        ("FONTNAME",     (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (1,0), (1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 9),
        ("TEXTCOLOR",    (0,0), (0,-1), NAVY),
        ("TEXTCOLOR",    (1,0), (1,-1), GREY),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[LGREY, WHITE]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#BDBDBD")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
    ])


# ════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ════════════════════════════════════════════════════════════
def generate_report(data, output_path="moodlens_report.pdf"):
    """
    Generate the MoodLens PDF report.

    Parameters
    ----------
    data : dict  — see SAMPLE_DATA at bottom for full schema
    output_path : str
    """
    S = make_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title="MoodLens Mood Analysis Report",
        author="MoodLens System"
    )

    story = []

    # ── COVER ─────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Mood Analysis Report", S["title"]))
    story.append(Paragraph("User Mood Monitoring System — MoodLens", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=NAVY, spaceAfter=12))

    # ── SECTION 1: USER INFORMATION ───────────────────────
    story.append(section_header("1. User Information", S))
    u = data["user"]
    info_rows = [
        ["User ID",              u.get("user_id", "—")],
        ["Report Period",        f"{u.get('period_start','—')} to {u.get('period_end','—')}"],
        ["Total Entries",        str(u.get("total_entries", "—"))],
        ["Model Used",           u.get("model", "XGBoost")],
        ["Questionnaire Version","Unified Mood Scale (30 items)"],
    ]
    t = Table(info_rows, colWidths=[5*cm, 11*cm])
    t.setStyle(kv_table_style())
    story.append(t)
    story.append(Spacer(1, 0.4*cm))

    # ── SECTION 2: OVERALL MOOD SUMMARY ───────────────────
    story.append(section_header("2. Overall Mood Summary", S))
    m = data["mood_summary"]
    pa  = m["avg_pa"]
    na  = m["avg_na"]
    bal = round(pa - na, 3)

    # Three metric boxes side by side
    metric_data = [[
        Paragraph(f"<b>{pa}</b>",  S["metric"]),
        Paragraph(f"<b>{na}</b>",  S["metric"]),
        Paragraph(f"<b>{bal:+.3f}</b>", S["metric"]),
    ],[
        Paragraph("Avg Positive Affect (PA)", S["label"]),
        Paragraph("Avg Negative Affect (NA)", S["label"]),
        Paragraph("Mood Balance (PA − NA)",   S["label"]),
    ]]
    mt = Table(metric_data, colWidths=[5.3*cm]*3)
    mt.setStyle(TableStyle([
        ("ALIGN",       (0,0),(-1,-1),"CENTER"),
        ("VALIGN",      (0,0),(-1,-1),"MIDDLE"),
        ("BACKGROUND",  (0,0),(-1,0), LBLUE),
        ("TOPPADDING",  (0,0),(-1,-1),8),
        ("BOTTOMPADDING",(0,0),(-1,-1),8),
        ("BOX",         (0,0),(-1,-1),0.8, BLUE),
        ("INNERGRID",   (0,0),(-1,-1),0.4, BLUE),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.3*cm))

    if bal > 0:
        interp = "Positive balance — overall positive mood state."
    elif bal < 0:
        interp = "Negative balance — emotional strain detected."
    else:
        interp = "Balanced — stable emotional state."

    story.append(Paragraph(
        f"Interpretation: {interp}  |  Overall Trend: {m.get('trend','—')}",
        S["interp"]
    ))

    # ── SECTION 3: WEEKLY MOOD TREND ──────────────────────
    story.append(section_header("3. Weekly Mood Trend", S))

    # Table
    weekly_rows = [["Week", "PA Score", "NA Score", "Mood Balance", "Label"]]
    for w in data["weekly"]:
        b = round(w["pa_score"] - w["na_score"], 3)
        weekly_rows.append([
            f"Week {w['week']}",
            f"{w['pa_score']:.3f}",
            f"{w['na_score']:.3f}",
            f"{b:+.3f}",
            w["mood_label"]
        ])
    wt = Table(weekly_rows, colWidths=[2.5*cm, 3*cm, 3*cm, 3.5*cm, 4*cm])
    wt.setStyle(header_table_style())
    story.append(wt)
    story.append(Spacer(1, 0.4*cm))

    # Line chart
    story.append(chart_pa_na_trend(data["weekly"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_mood_stability(data["weekly"]))
    story.append(Spacer(1, 0.3*cm))

    # ── SECTION 4: EMOTIONAL DIMENSIONS ──────────────────
    story.append(section_header("4. Emotional Dimension Analysis", S))

    story.append(Paragraph("Positive Affect Dimensions", S["subsection"]))
    pa_items = data["pa_dimensions"]
    pa_rows  = [["Item", "Avg Score"]] + [[k, f"{v:.3f}"] for k, v in pa_items.items()]
    pa_t = Table(pa_rows, colWidths=[8*cm, 8*cm])
    pa_t.setStyle(header_table_style())
    story.append(pa_t)
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_radar(pa_items, "Positive Affect Profile", mBLUE))

    story.append(Paragraph(
        data.get("pa_interpretation",
                 "Energy and motivation scores show gradual improvement across the reporting period."),
        S["interp"]
    ))

    story.append(Paragraph("Negative Affect Dimensions", S["subsection"]))
    na_items = data["na_dimensions"]
    na_rows  = [["Item", "Avg Score"]] + [[k, f"{v:.3f}"] for k, v in na_items.items()]
    na_t = Table(na_rows, colWidths=[8*cm, 8*cm])
    na_t.setStyle(header_table_style())
    story.append(na_t)
    story.append(Spacer(1, 0.3*cm))
    story.append(chart_radar(na_items, "Negative Affect Profile", mRED))

    story.append(Paragraph(
        data.get("na_interpretation",
                 "Anxiety-related indicators showed a declining trend after Week 2."),
        S["interp"]
    ))

    # ── SECTION 5: COGNITIVE INDICATORS ──────────────────
    story.append(section_header("5. Cognitive and Context Indicators", S))
    cog = data["cognitive_indicators"]
    cog_rows = [["Indicator", "Average Score"]] + [[k, f"{v:.3f}"] for k, v in cog.items()]
    cog_t = Table(cog_rows, colWidths=[9*cm, 7*cm])
    cog_t.setStyle(header_table_style())
    story.append(cog_t)
    story.append(Paragraph(
        data.get("cognitive_interpretation",
                 "Rumination was elevated in early weeks but declined steadily."),
        S["interp"]
    ))

    # ── SECTION 6: LIFESTYLE FACTORS ─────────────────────
    story.append(section_header("6. Lifestyle and Wellness Factors", S))
    lf = data["lifestyle_factors"]
    lf_rows = [["Factor", "Average Score"]] + [[k, f"{v:.3f}"] for k, v in lf.items()]
    lf_t = Table(lf_rows, colWidths=[9*cm, 7*cm])
    lf_t.setStyle(header_table_style())
    story.append(lf_t)
    story.append(Paragraph(
        data.get("lifestyle_interpretation",
                 "Lower sleep quality correlated with elevated negative affect scores."),
        S["interp"]
    ))

    # ── SECTION 7: EMOTIONAL PROFILE ─────────────────────
    story.append(section_header("7. Emotional Profile Visualisation", S))
    story.append(Paragraph(
        "The Plutchik Wheel below shows the relative strength of eight primary "
        "emotions averaged across the reporting period. Petal length corresponds "
        "to average intensity on the 0–4 normalised scale.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(chart_plutchik_wheel(data["plutchik"]))
    story.append(Paragraph(
        "The wheel highlights dominant emotions and overall emotional balance over time.",
        S["interp"]
    ))

    # ── SECTION 8: MOOD STABILITY ANALYSIS ───────────────
    story.append(section_header("8. Mood Stability Analysis", S))
    ms = data["stability"]
    stab_rows = [
        ["Mood Variability Index",  f"{ms.get('variability_index', '—')}"],
        ["Emotional Stability Score", f"{ms.get('stability_score', '—')}"],
        ["Label Distribution",      ms.get("label_distribution", "—")],
    ]
    st = Table(stab_rows, colWidths=[7*cm, 9*cm])
    st.setStyle(kv_table_style())
    story.append(st)
    vi = ms.get("variability_index", 0)
    if isinstance(vi, (int, float)):
        stab_interp = (
            "Low variability — stable mood across the reporting period."
            if vi < 0.5 else
            "High variability — fluctuating emotional state; consider monitoring closely."
        )
    else:
        stab_interp = ms.get("stability_interpretation", "—")
    story.append(Paragraph(stab_interp, S["interp"]))

    # ── SECTION 9: MODEL INSIGHTS ─────────────────────────
    story.append(section_header("9. Model Insights (SHAP Feature Importance)", S))
    story.append(Paragraph(
        "SHAP (SHapley Additive exPlanations) values quantify each feature's "
        "contribution to the model's prediction. Higher mean absolute SHAP value "
        "indicates greater influence on mood classification.",
        S["body"]
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(chart_shap_bar(data["shap_importance"]))
    story.append(Paragraph(
        data.get("shap_interpretation",
                 "Stress level and rumination were the strongest predictors of negative affect."),
        S["interp"]
    ))

    # ── SECTION 10: OBSERVED PATTERNS ────────────────────
    story.append(section_header("10. Observed Patterns", S))
    for pattern in data.get("observed_patterns", []):
        story.append(Paragraph(f"• {pattern}", S["body"]))
    story.append(Spacer(1, 0.3*cm))

    # ── SECTION 11: FINAL SUMMARY ────────────────────────
    story.append(section_header("11. Final Summary", S))
    final = data.get("final_summary", "No summary provided.")
    if isinstance(final, list):
        for para in final:
            story.append(Paragraph(para, S["body"]))
            story.append(Spacer(1, 0.25*cm))
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
    print(f"Report saved: {output_path}")


# ════════════════════════════════════════════════════════════
# SAMPLE DATA — replace with real model outputs
# ════════════════════════════════════════════════════════════
SAMPLE_DATA = {

    # ── Section 1 ─────────────────────────────────────────
    "user": {
        "user_id":       "ML-001",
        "period_start":  "2025-03-01",
        "period_end":    "2025-03-28",
        "total_entries": 4,
        "model":         "XGBoost (v6)",
    },

    # ── Section 2 ─────────────────────────────────────────
    "mood_summary": {
        "avg_pa":  2.847,
        "avg_na":  1.612,
        "trend":   "Improving",
    },

    # ── Section 3 ─────────────────────────────────────────
    "weekly": [
        {"week": 1, "pa_score": 2.10, "na_score": 2.30, "mood_label": "Unstable"},
        {"week": 2, "pa_score": 2.65, "na_score": 1.85, "mood_label": "Moderate"},
        {"week": 3, "pa_score": 3.10, "na_score": 1.40, "mood_label": "Stable"},
        {"week": 4, "pa_score": 3.32, "na_score": 0.90, "mood_label": "Stable"},
    ],

    # ── Section 4 ─────────────────────────────────────────
    "pa_dimensions": {
        "Interested":    2.75,
        "Excited":       2.20,
        "Enthusiastic":  2.50,
        "Proud":         2.80,
        "Strong":        2.65,
        "Inspired":      3.10,
        "Alert":         3.00,
        "Determined":    3.25,
        "Attentive":     2.90,
        "Active":        2.60,
    },
    "pa_interpretation": (
        "Energy and motivation scores increased gradually across the reporting period, "
        "with Determined and Inspired reaching the highest averages."
    ),
    "na_dimensions": {
        "Distressed":  1.80,
        "Upset":       2.10,
        "Guilty":      1.20,
        "Scared":      1.05,
        "Hostile":     0.90,
        "Irritable":   1.95,
        "Ashamed":     0.85,
        "Nervous":     1.60,
        "Jittery":     1.40,
        "Afraid":      1.00,
    },
    "na_interpretation": (
        "Anxiety-related indicators (Nervous, Jittery) decreased after Week 2. "
        "Upset and Irritable remained the dominant negative drivers."
    ),

    # ── Section 5 ─────────────────────────────────────────
    "cognitive_indicators": {
        "Emotional Clarity":    2.50,
        "Somatic Awareness":    2.10,
        "Rumination":           2.75,
        "Psychological Safety": 2.90,
        "Social Connection":    3.20,
    },
    "cognitive_interpretation": (
        "Rumination was elevated in early weeks but declined steadily from Week 2 onward. "
        "Social connection remained consistently high throughout the period."
    ),

    # ── Section 6 ─────────────────────────────────────────
    "lifestyle_factors": {
        "Sleep Quality":   2.40,
        "Stress Level":    2.85,
        "Focus Ability":   2.60,
        "Physical Fatigue":2.20,
    },
    "lifestyle_interpretation": (
        "Lower sleep quality in Weeks 1–2 correlated with higher negative affect. "
        "Stress level was the strongest lifestyle predictor across all four weeks."
    ),

    # ── Section 7 ─────────────────────────────────────────
    "plutchik": {
        "Joy":          3.20,
        "Trust":        2.90,
        "Fear":         1.60,
        "Surprise":     1.80,
        "Sadness":      1.40,
        "Disgust":      0.90,
        "Anger":        1.20,
        "Anticipation": 2.80,
    },

    # ── Section 8 ─────────────────────────────────────────
    "stability": {
        "variability_index":     0.38,
        "stability_score":       0.72,
        "label_distribution":    "Unstable × 1 | Moderate × 1 | Stable × 2",
        "stability_interpretation": (
            "Low variability — mood stabilised progressively across the four-week period."
        ),
    },

    # ── Section 9 ─────────────────────────────────────────
    "shap_importance": {
        "Q4_Upset":      0.0851,
        "Q11_Irritable": 0.0766,
        "Q2_Distressed": 0.0662,
        "Q9_Enthusiastic":0.0603,
        "Q1_Interested": 0.0509,
        "Q18_Jittery":   0.0191,
        "Q7_Scared":     0.0178,
        "Q15_Nervous":   0.0109,
        "Focus_Ability": 0.0008,
    },
    "shap_interpretation": (
        "Stress (Q4_Upset, Q11_Irritable) and distress (Q2_Distressed) were the strongest "
        "predictors of mood instability. Q9_Enthusiastic acted as a stabilising counter-signal."
    ),

    # ── Section 10 ────────────────────────────────────────
    "observed_patterns": [
        "Higher stress days in Weeks 1–2 corresponded with elevated negative affect scores.",
        "Improved sleep quality in Weeks 3–4 was associated with increased positive affect.",
        "Emotional clarity scores improved progressively across the four-week period.",
        "Social connection remained consistently high and correlated with mood stability.",
        "Rumination peaked in Week 1 and declined by approximately 40% by Week 4.",
    ],

    # ── Section 11 ────────────────────────────────────────
    "final_summary": (
        "Over the four-week monitoring period, the user's mood shifted from an Unstable "
        "state in Week 1 to two consecutive Stable readings in Weeks 3 and 4. "
        "Positive affect increased steadily while negative affect declined, producing "
        "an improving mood balance trajectory. Key drivers of the initial instability — "
        "elevated stress and rumination — both showed measurable decline from Week 2 "
        "onward. Sleep quality improvement coincided with the mood stabilisation observed "
        "from Week 3. The SHAP analysis confirms that upset and irritability were the "
        "dominant classification signals throughout the period. Overall, the data indicates "
        "a genuine and sustained improvement in emotional well-being across this reporting window."
    ),
}


# ════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    generate_report(SAMPLE_DATA, "moodlens_report.pdf")