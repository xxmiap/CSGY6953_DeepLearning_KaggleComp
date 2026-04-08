# SVG Generation with Qwen2.5 + LoRA Fine-tuning

## Project Overview

This project focuses on generating structured SVG code from natural language prompts using a fine-tuned large language model.

We build upon the Qwen2.5 model and apply parameter-efficient fine-tuning (LoRA) to adapt the model for SVG generation tasks. The goal is to produce syntactically valid and semantically accurate SVG outputs given textual descriptions.

---

## Methodology

### Model

* Base Model: Qwen2.5
* Fine-tuning: LoRA (Low-Rank Adaptation)
* Framework: PyTorch + HuggingFace Transformers

### Key Ideas

* Treat SVG generation as a structured text generation task
* Use instruction-style prompts to guide output format
* Apply LoRA to reduce training cost and memory usage

---

## Data

The dataset consists of:

* Natural language prompts describing visual elements
* Corresponding SVG code

### Preprocessing

* Cleaned malformed SVG
* Normalized attribute formats
* Removed inconsistent samples

---

##  Exploratory Data Analysis (EDA)

We performed several analyses to understand the dataset:

* Prompt length distribution
* SVG complexity distribution
* Tag frequency analysis
* Attribute usage patterns
* Prompt-SVG alignment scoring

Results and visualizations are available in:

```
eda_output/
```

---

##  Training Setup

* Optimizer: AdamW
* Training Strategy: LoRA fine-tuning
* Loss: Cross-entropy
* Hardware: GPU (recommended)

---

## Inference

The final model generates SVG code from text prompts.

**Example**

Input:

```
Draw a red circle with radius 50 at center (100,100)
```

Output:

```xml
<svg>
  <circle cx="100" cy="100" r="50" fill="red"/>
</svg>
```

---

##  Repository Structure

```
Midterm/
├── baseline-qwen2-5_final.ipynb   # Main notebook (training + inference)
├── data_description.md
├── eda.py
├── eda_output/
├── export_model.py
├── sample_submission.csv
```

---

## How to Run

1. Open:

```
baseline-qwen2-5_final.ipynb
```

2. Run all cells:

* Load model
* Run inference
* Generate submission

---

## Results

* The model generates valid SVG structures for most prompts
* Fine-tuning improves alignment between prompt and SVG output
* Performs well on basic geometric and attribute-based tasks

---

##  Limitations

* Complex SVG scenes may be partially incorrect
* Long outputs may lose precision
* Generalization is limited to training distribution

---

##  Model Weights

Model weights are provided via GitHub Releases:

👉 https://github.com/xxmiap/CSGY6953_DeepLearning_KaggleComp/releases/tag/v1.0-model-weights

---

##  Conclusion

This project demonstrates that:

* LLMs can generate structured graphical code effectively
* LoRA is efficient for domain-specific adaptation
* Data preprocessing plays a critical role in performance

---

## Author

Xiao Xiao
Shang Xu
