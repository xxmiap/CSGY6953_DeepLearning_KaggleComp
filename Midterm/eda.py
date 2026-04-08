#!/usr/bin/env python3
"""
Comprehensive EDA for Text-to-SVG Generation Task
NYU Deep Learning — Spring 2026 Midterm

Covers 5 modules:
  1. Basic Stats (data scale, duplicates, mappings)
  2. SVG Structure Analysis (tags, complexity, attributes)
  3. Length & Token Analysis (char/token lengths)
  4. Data Quality Checks (XML validity, anomalies, float precision)
  5. Prompt–SVG Alignment (keyword overlap heuristic)

Outputs:
  - Visualizations saved to eda_output/
  - Summary report saved to eda_output/eda_report.md
"""

import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Configuration ───────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
OUTPUT_DIR = DATA_DIR / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

report_lines: list[str] = []


def log(msg: str):
    print(msg)
    report_lines.append(msg)


def save_fig(name: str):
    path = OUTPUT_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  [saved] {path}")


# ─── Load Data ───────────────────────────────────────────────────────────────

log("=" * 70)
log("LOADING DATA")
log("=" * 70)

df = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

log(f"Train shape: {df.shape}")
log(f"Test  shape: {df_test.shape}")
log(f"Train columns: {list(df.columns)}")
log(f"Test  columns: {list(df_test.columns)}")
log("")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1: Basic Stats
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 1: BASIC STATS")
log("=" * 70)

n_samples = len(df)
n_unique_prompt = df["prompt"].nunique()
n_unique_svg = df["svg"].nunique()

log(f"Total samples:    {n_samples}")
log(f"Unique prompts:   {n_unique_prompt} ({n_unique_prompt/n_samples*100:.2f}%)")
log(f"Unique SVGs:      {n_unique_svg} ({n_unique_svg/n_samples*100:.2f}%)")
log("")

# Duplicates
dup_rows = df.duplicated().sum()
dup_prompt = df["prompt"].duplicated().sum()
dup_svg = df["svg"].duplicated().sum()

log(f"Duplicate rows:    {dup_rows} ({dup_rows/n_samples*100:.2f}%)")
log(f"Duplicate prompts: {dup_prompt} ({dup_prompt/n_samples*100:.2f}%)")
log(f"Duplicate SVGs:    {dup_svg} ({dup_svg/n_samples*100:.2f}%)")
log("")

# NaN check
nan_prompt = df["prompt"].isna().sum()
nan_svg = df["svg"].isna().sum()
log(f"NaN prompts: {nan_prompt}")
log(f"NaN SVGs:    {nan_svg}")
log("")

# Prompt–SVG mapping: how many prompts per unique SVG?
svg_prompt_count = df.groupby("svg")["prompt"].nunique()
log("Prompts per unique SVG:")
log(f"  mean:   {svg_prompt_count.mean():.2f}")
log(f"  median: {svg_prompt_count.median():.0f}")
log(f"  max:    {svg_prompt_count.max()}")
log(f"  SVGs with >1 prompt: {(svg_prompt_count > 1).sum()}")
log("")

# Prompt per SVG distribution
fig, ax = plt.subplots(figsize=(8, 5))
svg_prompt_count.clip(upper=10).value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_xlabel("Number of unique prompts per SVG")
ax.set_ylabel("Count of SVGs")
ax.set_title("Prompt-per-SVG Mapping Distribution (clipped at 10)")
save_fig("1_prompt_per_svg")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2: SVG Structure Analysis
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 2: SVG STRUCTURE ANALYSIS")
log("=" * 70)

SVG_TAGS = [
    "path", "circle", "rect", "line", "ellipse", "polygon", "polyline",
    "text", "g", "use", "defs", "clipPath", "mask", "linearGradient",
    "radialGradient", "stop", "image", "pattern", "symbol",
]

tag_counter = Counter()
per_sample_complexity = []
per_sample_tags = []

has_viewBox = 0
has_width_height = 0
has_fill = 0
has_stroke = 0
has_transform = 0
has_style = 0

parse_failures_struct = 0

for idx, svg_str in enumerate(df["svg"]):
    if pd.isna(svg_str) or not isinstance(svg_str, str):
        per_sample_complexity.append(0)
        per_sample_tags.append({})
        continue
    try:
        root = ET.fromstring(svg_str)
    except ET.ParseError:
        parse_failures_struct += 0  # counted in module 4
        per_sample_complexity.append(0)
        per_sample_tags.append({})
        continue

    ns = {"svg": "http://www.w3.org/2000/svg"}
    all_elems = list(root.iter())
    n_elems = len(all_elems) - 1  # exclude <svg> root itself if present

    local_tags = Counter()
    for elem in all_elems:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        local_tags[tag] += 1
        tag_counter[tag] += 1

    per_sample_complexity.append(n_elems)
    per_sample_tags.append(dict(local_tags))

    root_attribs = root.attrib
    svg_text_lower = svg_str.lower()
    if "viewbox" in {k.lower() for k in root_attribs}:
        has_viewBox += 1
    if "width" in root_attribs or "height" in root_attribs:
        has_width_height += 1
    if "fill" in svg_text_lower:
        has_fill += 1
    if "stroke" in svg_text_lower:
        has_stroke += 1
    if "transform" in svg_text_lower:
        has_transform += 1
    if "<style" in svg_text_lower or 'style="' in svg_text_lower:
        has_style += 1

df["complexity"] = per_sample_complexity

# Tag distribution (top 15)
top_tags = tag_counter.most_common(20)
log("Top SVG tags (across all samples):")
for tag, cnt in top_tags:
    log(f"  {tag:20s}: {cnt:>8d}")
log("")

fig, ax = plt.subplots(figsize=(10, 6))
tags, counts = zip(*top_tags)
ax.barh(tags[::-1], counts[::-1])
ax.set_xlabel("Total Occurrences")
ax.set_title("SVG Tag Distribution (Top 20)")
save_fig("2_tag_distribution")

# Complexity distribution
log("SVG Complexity (number of child elements):")
log(f"  mean:   {df['complexity'].mean():.2f}")
log(f"  median: {df['complexity'].median():.0f}")
log(f"  std:    {df['complexity'].std():.2f}")
log(f"  min:    {df['complexity'].min()}")
log(f"  max:    {df['complexity'].max()}")
log(f"  p95:    {df['complexity'].quantile(0.95):.0f}")
log(f"  p99:    {df['complexity'].quantile(0.99):.0f}")
log("")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["complexity"], bins=80, edgecolor="black", alpha=0.7)
axes[0].set_xlabel("Number of Elements")
axes[0].set_ylabel("Count")
axes[0].set_title("SVG Complexity Distribution")
axes[0].axvline(df["complexity"].quantile(0.95), color="red", linestyle="--", label="p95")
axes[0].axvline(df["complexity"].quantile(0.99), color="orange", linestyle="--", label="p99")
axes[0].legend()

axes[1].hist(df["complexity"].clip(upper=df["complexity"].quantile(0.99)),
             bins=60, edgecolor="black", alpha=0.7)
axes[1].set_xlabel("Number of Elements (clipped at p99)")
axes[1].set_ylabel("Count")
axes[1].set_title("SVG Complexity Distribution (zoomed)")
save_fig("2_complexity_distribution")

# Attribute usage
log("SVG Attribute Usage:")
log(f"  viewBox:       {has_viewBox:>6d} ({has_viewBox/n_samples*100:.1f}%)")
log(f"  width/height:  {has_width_height:>6d} ({has_width_height/n_samples*100:.1f}%)")
log(f"  fill:          {has_fill:>6d} ({has_fill/n_samples*100:.1f}%)")
log(f"  stroke:        {has_stroke:>6d} ({has_stroke/n_samples*100:.1f}%)")
log(f"  transform:     {has_transform:>6d} ({has_transform/n_samples*100:.1f}%)")
log(f"  style:         {has_style:>6d} ({has_style/n_samples*100:.1f}%)")
log("")

fig, ax = plt.subplots(figsize=(8, 4))
attr_names = ["viewBox", "width/height", "fill", "stroke", "transform", "style"]
attr_vals = [has_viewBox, has_width_height, has_fill, has_stroke, has_transform, has_style]
attr_pcts = [v / n_samples * 100 for v in attr_vals]
bars = ax.barh(attr_names, attr_pcts)
ax.set_xlabel("Percentage of Samples (%)")
ax.set_title("SVG Attribute Usage")
for bar, pct in zip(bars, attr_pcts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center")
save_fig("2_attribute_usage")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3: Length & Token Analysis
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 3: LENGTH & TOKEN ANALYSIS")
log("=" * 70)

df["prompt_len"] = df["prompt"].fillna("").str.len()
df["svg_len"] = df["svg"].fillna("").str.len()
df["prompt_word_count"] = df["prompt"].fillna("").str.split().str.len()

log("Prompt character length:")
log(f"  mean:   {df['prompt_len'].mean():.1f}")
log(f"  median: {df['prompt_len'].median():.0f}")
log(f"  min:    {df['prompt_len'].min()}")
log(f"  max:    {df['prompt_len'].max()}")
log(f"  p95:    {df['prompt_len'].quantile(0.95):.0f}")
log(f"  p99:    {df['prompt_len'].quantile(0.99):.0f}")
log("")

log("Prompt word count:")
log(f"  mean:   {df['prompt_word_count'].mean():.1f}")
log(f"  median: {df['prompt_word_count'].median():.0f}")
log(f"  min:    {df['prompt_word_count'].min()}")
log(f"  max:    {df['prompt_word_count'].max()}")
log("")

log("SVG character length:")
log(f"  mean:   {df['svg_len'].mean():.1f}")
log(f"  median: {df['svg_len'].median():.0f}")
log(f"  min:    {df['svg_len'].min()}")
log(f"  max:    {df['svg_len'].max()}")
log(f"  p95:    {df['svg_len'].quantile(0.95):.0f}")
log(f"  p99:    {df['svg_len'].quantile(0.99):.0f}")
log("")

# Token estimation (rough: 1 token ≈ 4 chars for English, ~3.5 for code/XML)
df["svg_token_est"] = (df["svg_len"] / 3.5).astype(int)
df["prompt_token_est"] = (df["prompt_len"] / 4.0).astype(int)

log("Estimated token counts (svg_len/3.5, prompt_len/4.0):")
log(f"  SVG  tokens — mean: {df['svg_token_est'].mean():.0f}, "
    f"median: {df['svg_token_est'].median():.0f}, "
    f"max: {df['svg_token_est'].max()}")
log(f"  Prompt tokens — mean: {df['prompt_token_est'].mean():.0f}, "
    f"median: {df['prompt_token_est'].median():.0f}, "
    f"max: {df['prompt_token_est'].max()}")
log("")

# Context window analysis
for limit in [2048, 4096, 8192]:
    exceed = (df["svg_token_est"] > limit).sum()
    log(f"  SVG exceeding ~{limit} tokens: {exceed} ({exceed/n_samples*100:.2f}%)")
log("")

# Prompt length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["prompt_len"], bins=60, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].set_xlabel("Prompt Character Length")
axes[0].set_ylabel("Count")
axes[0].set_title("Prompt Length Distribution")

axes[1].hist(df["prompt_word_count"], bins=40, edgecolor="black", alpha=0.7, color="teal")
axes[1].set_xlabel("Prompt Word Count")
axes[1].set_ylabel("Count")
axes[1].set_title("Prompt Word Count Distribution")
save_fig("3_prompt_length")

# SVG length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["svg_len"], bins=80, edgecolor="black", alpha=0.7, color="coral")
axes[0].set_xlabel("SVG Character Length")
axes[0].set_ylabel("Count")
axes[0].set_title("SVG Length Distribution")
axes[0].axvline(df["svg_len"].quantile(0.95), color="red", linestyle="--", label="p95")
axes[0].axvline(df["svg_len"].quantile(0.99), color="orange", linestyle="--", label="p99")
axes[0].legend()

axes[1].hist(df["svg_len"].clip(upper=df["svg_len"].quantile(0.99)),
             bins=60, edgecolor="black", alpha=0.7, color="salmon")
axes[1].set_xlabel("SVG Length (clipped at p99)")
axes[1].set_ylabel("Count")
axes[1].set_title("SVG Length Distribution (zoomed)")
save_fig("3_svg_length")

# Joint plot: prompt_len vs svg_len
fig, ax = plt.subplots(figsize=(8, 8))
sample_idx = np.random.choice(len(df), size=min(5000, len(df)), replace=False)
ax.scatter(df.loc[sample_idx, "prompt_len"],
           df.loc[sample_idx, "svg_len"],
           alpha=0.15, s=8, color="purple")
ax.set_xlabel("Prompt Character Length")
ax.set_ylabel("SVG Character Length")
ax.set_title("Prompt Length vs SVG Length (sampled 5k)")
save_fig("3_prompt_vs_svg_length")

# SVG length by complexity
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df["complexity"], df["svg_len"], alpha=0.1, s=5, color="teal")
ax.set_xlabel("SVG Complexity (# elements)")
ax.set_ylabel("SVG Character Length")
ax.set_title("Complexity vs SVG Length")
save_fig("3_complexity_vs_length")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4: Data Quality Checks
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 4: DATA QUALITY CHECKS")
log("=" * 70)

# 4.1 XML parse validity
parse_ok = 0
parse_fail = 0
parse_fail_indices = []

for idx, svg_str in enumerate(df["svg"]):
    if pd.isna(svg_str) or not isinstance(svg_str, str):
        parse_fail += 1
        parse_fail_indices.append(idx)
        continue
    try:
        ET.fromstring(svg_str)
        parse_ok += 1
    except ET.ParseError:
        parse_fail += 1
        parse_fail_indices.append(idx)

log(f"XML Parse Success: {parse_ok} ({parse_ok/n_samples*100:.2f}%)")
log(f"XML Parse Failure: {parse_fail} ({parse_fail/n_samples*100:.2f}%)")
if parse_fail_indices:
    log(f"  Failed indices (first 10): {parse_fail_indices[:10]}")
log("")

# 4.2 Empty / extremely short SVGs
empty_svg = (df["svg_len"] == 0).sum()
very_short_svg = (df["svg_len"] < 50).sum()
very_long_svg = (df["svg_len"] > df["svg_len"].quantile(0.99)).sum()

log(f"Empty SVGs:          {empty_svg}")
log(f"Very short SVGs (<50 chars): {very_short_svg}")
log(f"Very long SVGs (>p99):       {very_long_svg}")
log("")

empty_prompt = (df["prompt_len"] == 0).sum()
very_short_prompt = (df["prompt_len"] < 5).sum()
log(f"Empty prompts:              {empty_prompt}")
log(f"Very short prompts (<5):    {very_short_prompt}")
log("")

# 4.3 Float precision analysis
float_pattern = re.compile(r"-?\d+\.\d+")
decimal_lengths = []
sample_for_float = df["svg"].dropna().sample(min(5000, n_samples), random_state=42)

for svg_str in sample_for_float:
    floats = float_pattern.findall(svg_str)
    for f in floats:
        decimal_part = f.split(".")[-1]
        decimal_lengths.append(len(decimal_part))

if decimal_lengths:
    dec_counter = Counter(decimal_lengths)
    log("Float decimal precision distribution (sampled):")
    for prec in sorted(dec_counter.keys()):
        pct = dec_counter[prec] / len(decimal_lengths) * 100
        log(f"  {prec} decimal places: {dec_counter[prec]:>8d} ({pct:.1f}%)")
    log(f"  Total floats sampled: {len(decimal_lengths)}")
    log("")

    fig, ax = plt.subplots(figsize=(8, 5))
    precs = sorted(dec_counter.keys())
    vals = [dec_counter[p] for p in precs]
    ax.bar(precs, vals, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Decimal Places")
    ax.set_ylabel("Count")
    ax.set_title("Float Precision Distribution in SVGs (sampled)")
    ax.set_xticks(precs)
    save_fig("4_float_precision")

# 4.4 SVG root tag check
non_svg_root = 0
for svg_str in df["svg"].dropna():
    try:
        root = ET.fromstring(svg_str)
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag != "svg":
            non_svg_root += 1
    except ET.ParseError:
        pass

log(f"SVGs with non-<svg> root tag: {non_svg_root}")
log("")

# 4.5 viewBox value analysis
viewbox_values = []
for svg_str in df["svg"].dropna():
    try:
        root = ET.fromstring(svg_str)
        vb = root.attrib.get("viewBox") or root.attrib.get("viewbox", "")
        if vb:
            viewbox_values.append(vb.strip())
    except ET.ParseError:
        pass

if viewbox_values:
    vb_counter = Counter(viewbox_values)
    log(f"Unique viewBox values: {len(vb_counter)}")
    log("Top 10 viewBox values:")
    for vb, cnt in vb_counter.most_common(10):
        log(f"  '{vb}': {cnt}")
    log("")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5: Prompt–SVG Alignment (Heuristic)
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 5: PROMPT–SVG ALIGNMENT (Heuristic)")
log("=" * 70)

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "cyan", "magenta", "teal", "navy",
    "gold", "silver", "maroon", "olive", "lime", "aqua", "beige", "ivory",
    "coral", "salmon", "khaki", "indigo", "violet", "turquoise",
}

SHAPE_WORDS = {
    "circle", "square", "rectangle", "triangle", "star", "heart", "diamond",
    "oval", "ellipse", "hexagon", "pentagon", "arrow", "line", "cross",
    "ring", "arc", "curve", "polygon", "spiral",
}

SVG_SHAPE_TAGS = {"circle", "rect", "ellipse", "line", "polygon", "polyline", "path"}

def keyword_overlap_score(prompt_text: str, svg_text: str) -> dict:
    """Simple heuristic: check if color/shape words in prompt appear in SVG."""
    prompt_lower = prompt_text.lower()
    svg_lower = svg_text.lower()
    prompt_words = set(prompt_lower.split())

    colors_in_prompt = prompt_words & COLOR_WORDS
    shapes_in_prompt = prompt_words & SHAPE_WORDS

    color_hits = sum(1 for c in colors_in_prompt if c in svg_lower)
    shape_hits = sum(1 for s in shapes_in_prompt
                     if s in svg_lower or any(t in svg_lower for t in SVG_SHAPE_TAGS if s.startswith(t[:4])))

    total_kw = len(colors_in_prompt) + len(shapes_in_prompt)
    total_hits = color_hits + shape_hits
    score = total_hits / total_kw if total_kw > 0 else np.nan

    return {
        "colors_in_prompt": len(colors_in_prompt),
        "shapes_in_prompt": len(shapes_in_prompt),
        "color_hits": color_hits,
        "shape_hits": shape_hits,
        "total_keywords": total_kw,
        "total_hits": total_hits,
        "score": score,
    }


alignment_results = []
for _, row in df.iterrows():
    if pd.isna(row["prompt"]) or pd.isna(row["svg"]):
        alignment_results.append({"score": np.nan})
        continue
    alignment_results.append(keyword_overlap_score(row["prompt"], row["svg"]))

df_align = pd.DataFrame(alignment_results)

has_kw = df_align["total_keywords"] > 0
log(f"Samples with detectable keywords in prompt: {has_kw.sum()} ({has_kw.sum()/n_samples*100:.1f}%)")
log("")

valid_scores = df_align.loc[has_kw, "score"]
if len(valid_scores) > 0:
    log("Keyword overlap score (among samples with keywords):")
    log(f"  mean:   {valid_scores.mean():.3f}")
    log(f"  median: {valid_scores.median():.3f}")
    log(f"  score=0 (no overlap): {(valid_scores == 0).sum()} ({(valid_scores==0).sum()/len(valid_scores)*100:.1f}%)")
    log(f"  score=1 (full overlap): {(valid_scores == 1).sum()} ({(valid_scores==1).sum()/len(valid_scores)*100:.1f}%)")
    log("")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(valid_scores, bins=20, edgecolor="black", alpha=0.7, color="mediumpurple")
    ax.set_xlabel("Keyword Overlap Score")
    ax.set_ylabel("Count")
    ax.set_title("Prompt–SVG Keyword Overlap Score Distribution")
    save_fig("5_alignment_score")

# Low-score samples
low_score_mask = (df_align["score"] < 0.3) & (df_align["total_keywords"] >= 2)
low_score_count = low_score_mask.sum()
log(f"Potentially misaligned samples (score<0.3, >=2 keywords): {low_score_count}")
if low_score_count > 0:
    low_examples = df.loc[low_score_mask].head(5)
    log("  Example misaligned prompts:")
    for _, row in low_examples.iterrows():
        log(f"    - \"{row['prompt'][:100]}...\"")
log("")

# Prompt topic analysis (most frequent words)
all_words = " ".join(df["prompt"].dropna()).lower().split()
stop_words = {"the", "a", "an", "is", "in", "on", "of", "and", "with", "to", "it",
              "its", "that", "this", "for", "are", "at", "by", "from", "as", "be",
              "was", "or", "has", "have", "had", "not", "but", "which", "each",
              "set", "against", "features", "image", "two", "three", "one",
              "background", "depicting"}
word_freq = Counter(w for w in all_words if w not in stop_words and len(w) > 2)

log("Top 30 words in prompts (excluding stopwords):")
for word, cnt in word_freq.most_common(30):
    log(f"  {word:20s}: {cnt:>6d}")
log("")

fig, ax = plt.subplots(figsize=(10, 7))
top30 = word_freq.most_common(30)
words, freqs = zip(*top30)
ax.barh(words[::-1], freqs[::-1], color="steelblue")
ax.set_xlabel("Frequency")
ax.set_title("Top 30 Prompt Words")
save_fig("5_prompt_word_freq")

# ═════════════════════════════════════════════════════════════════════════════
# MODULE 6: Test Set Comparison
# ═════════════════════════════════════════════════════════════════════════════

log("=" * 70)
log("MODULE 6: TEST SET COMPARISON")
log("=" * 70)

df_test["prompt_len"] = df_test["prompt"].fillna("").str.len()
df_test["prompt_word_count"] = df_test["prompt"].fillna("").str.split().str.len()

log(f"Test prompts: {len(df_test)}")
log(f"Test prompt length — mean: {df_test['prompt_len'].mean():.1f}, "
    f"median: {df_test['prompt_len'].median():.0f}, "
    f"max: {df_test['prompt_len'].max()}")
log(f"Test prompt words  — mean: {df_test['prompt_word_count'].mean():.1f}, "
    f"median: {df_test['prompt_word_count'].median():.0f}")
log("")

# Overlap
test_prompts_in_train = df_test["prompt"].isin(df["prompt"]).sum()
log(f"Test prompts also in train: {test_prompts_in_train} ({test_prompts_in_train/len(df_test)*100:.1f}%)")
log("")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["prompt_len"], bins=50, alpha=0.5, label="Train", color="steelblue")
axes[0].hist(df_test["prompt_len"], bins=50, alpha=0.5, label="Test", color="coral")
axes[0].set_xlabel("Prompt Character Length")
axes[0].set_ylabel("Count")
axes[0].set_title("Prompt Length: Train vs Test")
axes[0].legend()

axes[1].hist(df["prompt_word_count"], bins=40, alpha=0.5, label="Train", color="steelblue")
axes[1].hist(df_test["prompt_word_count"], bins=40, alpha=0.5, label="Test", color="coral")
axes[1].set_xlabel("Prompt Word Count")
axes[1].set_ylabel("Count")
axes[1].set_title("Prompt Word Count: Train vs Test")
axes[1].legend()
save_fig("6_train_vs_test_prompt")

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY REPORT & CLEANING RECOMMENDATIONS
# ═════════════════════════════════════════════════════════════════════════════

log("")
log("=" * 70)
log("SUMMARY & CLEANING RECOMMENDATIONS")
log("=" * 70)
log("")
log("1. DATA SCALE & DUPLICATES")
log(f"   - {n_samples} training samples, {len(df_test)} test samples")
log(f"   - Duplicate rows: {dup_rows} ({dup_rows/n_samples*100:.1f}%)")
log(f"   - Duplicate prompts: {dup_prompt} ({dup_prompt/n_samples*100:.1f}%)")
log(f"   - Duplicate SVGs: {dup_svg} ({dup_svg/n_samples*100:.1f}%)")
log(f"   - SVGs with multiple prompts: {(svg_prompt_count > 1).sum()}")
log("")
log("2. SVG STRUCTURE")
log(f"   - Most common tags: {', '.join(t for t, _ in top_tags[:5])}")
log(f"   - Mean complexity: {df['complexity'].mean():.1f} elements")
log(f"   - Max complexity: {df['complexity'].max()} elements")
log(f"   - viewBox usage: {has_viewBox/n_samples*100:.1f}%")
log(f"   - fill usage: {has_fill/n_samples*100:.1f}%")
log("")
log("3. LENGTH & TOKEN")
log(f"   - SVG char length: mean={df['svg_len'].mean():.0f}, "
    f"median={df['svg_len'].median():.0f}, max={df['svg_len'].max()}")
log(f"   - Estimated SVG tokens: mean={df['svg_token_est'].mean():.0f}, "
    f"max={df['svg_token_est'].max()}")
log(f"   - SVGs > ~4096 tokens: {(df['svg_token_est'] > 4096).sum()}")
log("")
log("4. DATA QUALITY")
log(f"   - XML parse failures: {parse_fail} ({parse_fail/n_samples*100:.2f}%)")
log(f"   - NaN values: prompts={nan_prompt}, SVGs={nan_svg}")
log(f"   - Empty SVGs: {empty_svg}")
log(f"   - Non-<svg> root: {non_svg_root}")
log("")
log("5. CLEANING RECOMMENDATIONS")
if parse_fail > 0:
    log(f"   - REMOVE {parse_fail} samples with invalid XML ({parse_fail/n_samples*100:.2f}%)")
if nan_svg > 0 or nan_prompt > 0:
    log(f"   - REMOVE samples with NaN values (prompt={nan_prompt}, svg={nan_svg})")
if empty_svg > 0:
    log(f"   - REMOVE {empty_svg} empty SVG samples")
if dup_rows > 0:
    log(f"   - DEDUPLICATE {dup_rows} exact duplicate rows")

p99_svg_len = df["svg_len"].quantile(0.99)
top1_count = (df["svg_len"] > p99_svg_len).sum()
log(f"   - CONSIDER truncating/filtering top 1% longest SVGs (>{p99_svg_len:.0f} chars, {top1_count} samples)")

if decimal_lengths:
    most_common_prec = max(dec_counter, key=dec_counter.get)
    high_prec = sum(v for k, v in dec_counter.items() if k > 6)
    if high_prec > 0:
        log(f"   - CONSIDER rounding floats to <=6 decimal places (reduces token count)")

log(f"   - KEEP multi-prompt → single-SVG mappings (data augmentation)")

if has_viewBox < n_samples:
    missing_vb = n_samples - has_viewBox
    log(f"   - STANDARDIZE: {missing_vb} samples missing viewBox ({missing_vb/n_samples*100:.1f}%)")

log("")
log("=" * 70)
log("EDA COMPLETE. Visualizations saved to: " + str(OUTPUT_DIR))
log("=" * 70)

# ─── Save Report ─────────────────────────────────────────────────────────────

report_path = OUTPUT_DIR / "eda_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("# EDA Report — Text-to-SVG Generation\n\n")
    f.write("```\n")
    f.write("\n".join(report_lines))
    f.write("\n```\n")

print(f"\nReport saved to: {report_path}")
