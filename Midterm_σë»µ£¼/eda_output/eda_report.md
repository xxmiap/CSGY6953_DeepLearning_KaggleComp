# EDA Report — Text-to-SVG Generation

```
======================================================================
LOADING DATA
======================================================================
Train shape: (50000, 3)
Test  shape: (1000, 2)
Train columns: ['id', 'prompt', 'svg']
Test  columns: ['id', 'prompt']

======================================================================
MODULE 1: BASIC STATS
======================================================================
Total samples:    50000
Unique prompts:   45931 (91.86%)
Unique SVGs:      49993 (99.99%)

Duplicate rows:    0 (0.00%)
Duplicate prompts: 4069 (8.14%)
Duplicate SVGs:    7 (0.01%)

NaN prompts: 0
NaN SVGs:    0

Prompts per unique SVG:
  mean:   1.00
  median: 1
  max:    2
  SVGs with >1 prompt: 7

======================================================================
MODULE 2: SVG STRUCTURE ANALYSIS
======================================================================
Top SVG tags (across all samples):
  path                :   121005
  svg                 :    50000
  circle              :    13549
  g                   :     6101
  rect                :     4323
  stop                :     2821
  ellipse             :      773
  radialGradient      :      682
  linearGradient      :      589
  line                :      572
  clipPath            :      391
  defs                :      343
  text                :      304
  polygon             :      120
  title               :      118
  use                 :       75
  tspan               :       54
  style               :       48
  desc                :       33
  mask                :       26

SVG Complexity (number of child elements):
  mean:   3.04
  median: 2
  std:    4.79
  min:    0
  max:    306
  p95:    10
  p99:    20

SVG Attribute Usage:
  viewBox:        49917 (99.8%)
  width/height:   49588 (99.2%)
  fill:           48208 (96.4%)
  stroke:          3151 (6.3%)
  transform:       1998 (4.0%)
  style:            419 (0.8%)

======================================================================
MODULE 3: LENGTH & TOKEN ANALYSIS
======================================================================
Prompt character length:
  mean:   116.6
  median: 103
  min:    5
  max:    860
  p95:    252
  p99:    353

Prompt word count:
  mean:   19.7
  median: 17
  min:    1
  max:    127

SVG character length:
  mean:   2524.3
  median: 2110
  min:    91
  max:    15937
  p95:    6078
  p99:    7514

Estimated token counts (svg_len/3.5, prompt_len/4.0):
  SVG  tokens — mean: 721, median: 602, max: 4553
  Prompt tokens — mean: 29, median: 25, max: 215

  SVG exceeding ~2048 tokens: 862 (1.72%)
  SVG exceeding ~4096 tokens: 11 (0.02%)
  SVG exceeding ~8192 tokens: 0 (0.00%)

======================================================================
MODULE 4: DATA QUALITY CHECKS
======================================================================
XML Parse Success: 50000 (100.00%)
XML Parse Failure: 0 (0.00%)

Empty SVGs:          0
Very short SVGs (<50 chars): 0
Very long SVGs (>p99):       500

Empty prompts:              0
Very short prompts (<5):    0

Float decimal precision distribution (sampled):
  1 decimal places:   142196 (17.9%)
  2 decimal places:    90199 (11.4%)
  3 decimal places:    13578 (1.7%)
  4 decimal places:     6082 (0.8%)
  5 decimal places:     7434 (0.9%)
  6 decimal places:     9433 (1.2%)
  7 decimal places:    16415 (2.1%)
  8 decimal places:    26316 (3.3%)
  9 decimal places:    13788 (1.7%)
  10 decimal places:    12524 (1.6%)
  11 decimal places:     1511 (0.2%)
  12 decimal places:     4830 (0.6%)
  13 decimal places:    81196 (10.2%)
  14 decimal places:   283941 (35.8%)
  15 decimal places:    77272 (9.8%)
  16 decimal places:     5023 (0.6%)
  17 decimal places:      529 (0.1%)
  Total floats sampled: 792267

SVGs with non-<svg> root tag: 0

Unique viewBox values: 464
Top 10 viewBox values:
  '0.0 0.0 200.0 200.0': 40892
  '0 0 24 24': 2104
  '0 0 128 128': 902
  '0 0 400 400': 734
  '0 0 32 32': 655
  '0 0 512 512': 494
  '0 0 200 200': 491
  '0 0 100 100': 469
  '0 0 48 48': 373
  '0 0 64 64': 304

======================================================================
MODULE 5: PROMPT–SVG ALIGNMENT (Heuristic)
======================================================================
Samples with detectable keywords in prompt: 43788 (87.6%)

Keyword overlap score (among samples with keywords):
  mean:   0.030
  median: 0.000
  score=0 (no overlap): 42115 (96.2%)
  score=1 (full overlap): 1007 (2.3%)

Potentially misaligned samples (score<0.3, >=2 keywords): 32679
  Example misaligned prompts:
    - "The image features two orange squares with a microphone icon and an arrow connecting them, set again..."
    - "The image displays a black icon with a photo-like rectangle containing a wavy line on the left side ..."
    - "Generate svg code for an image that looks like: a blue icon with a white arrow. Don't use markdown j..."
    - "The image features a teal circular outline containing a solid teal star at its center, set against a..."
    - "A gray square icon with the text "MOV" and a small corner piece resembling a document...."

Top 30 words in prompts (excluding stopwords):
  black               :  32720
  white               :  30666
  background.         :  18600
  icon                :  17865
  circular            :   8219
  blue                :   7024
  shape               :   6663
  simple              :   6577
  line                :   6495
  horizontal          :   5616
  shows               :   5609
  code                :   5476
  lines               :   5457
  svg                 :   5439
  rectangular         :   5307
  outline             :   5258
  stylized            :   5204
  inside              :   4776
  single              :   4704
  gray                :   4571
  arrow               :   4520
  featuring           :   4421
  pointing            :   4020
  symbol              :   4011
  within              :   3964
  square              :   3910
  positioned          :   3811
  centered            :   3654
  it.                 :   3641
  contains            :   3480

======================================================================
MODULE 6: TEST SET COMPARISON
======================================================================
Test prompts: 1000
Test prompt length — mean: 119.3, median: 104, max: 488
Test prompt words  — mean: 20.1, median: 17

Test prompts also in train: 81 (8.1%)


======================================================================
SUMMARY & CLEANING RECOMMENDATIONS
======================================================================

1. DATA SCALE & DUPLICATES
   - 50000 training samples, 1000 test samples
   - Duplicate rows: 0 (0.0%)
   - Duplicate prompts: 4069 (8.1%)
   - Duplicate SVGs: 7 (0.0%)
   - SVGs with multiple prompts: 7

2. SVG STRUCTURE
   - Most common tags: path, svg, circle, g, rect
   - Mean complexity: 3.0 elements
   - Max complexity: 306 elements
   - viewBox usage: 99.8%
   - fill usage: 96.4%

3. LENGTH & TOKEN
   - SVG char length: mean=2524, median=2110, max=15937
   - Estimated SVG tokens: mean=721, max=4553
   - SVGs > ~4096 tokens: 11

4. DATA QUALITY
   - XML parse failures: 0 (0.00%)
   - NaN values: prompts=0, SVGs=0
   - Empty SVGs: 0
   - Non-<svg> root: 0

5. CLEANING RECOMMENDATIONS
   - CONSIDER truncating/filtering top 1% longest SVGs (>7514 chars, 500 samples)
   - CONSIDER rounding floats to <=6 decimal places (reduces token count)
   - KEEP multi-prompt → single-SVG mappings (data augmentation)
   - STANDARDIZE: 83 samples missing viewBox (0.2%)

======================================================================
EDA COMPLETE. Visualizations saved to: c:\Users\xxiao\OneDrive\桌面\Midterm_副本\Midterm_σë»µ£¼\eda_output
======================================================================
```
