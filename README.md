# Mechanistic Interpretability of Transformer Models

Investigating internal attention mechanisms of transformer architectures to understand **token influence** and **model decision pathways**. This project implements attention visualization, gradient-based feature attribution, LIME explanations, and layer-wise probing classifiers.

---

## Project Structure

```
mechanistic-interpretability/
│
├── src/
│   ├── attention_visualization.py   # Attention weight extraction & heatmaps
│   ├── feature_attribution.py       # Integrated Gradients + LIME attribution
│   ├── probing_classifiers.py       # Layer-wise probing experiments
│   └── __init__.py
│
├── notebooks/
│   └── demo.ipynb                   # End-to-end walkthrough
│
├── results/                         # Saved plots (auto-generated)
├── requirements.txt
└── README.md
```

---

## Features

| Module | Description |
|--------|-------------|
| `AttentionVisualizer` | Heatmaps per layer/head, token influence bar plots, multi-head diversity grids |
| `GradientAttribution` | Integrated Gradients — token-level importance via embedding gradients |
| `LIMETextExplainer` | LIME perturbation-based local explanation for classification models |
| `ProbingExperiment` | Logistic regression probes on each layer's CLS/mean-pooled representations |

---

## Setup

```bash
git clone https://github.com/trinadh3nadh/mechanistic-interpretability.git
cd mechanistic-interpretability
pip install -r requirements.txt
```

---

## Quick Start

### Attention Visualization

```python
from src.attention_visualization import AttentionExtractor, AttentionVisualizer

extractor = AttentionExtractor("bert-base-uncased")
visualizer = AttentionVisualizer(extractor)

text = "The cat sat on the mat near the warm fireplace."

# Average attention heatmap
visualizer.plot_attention_heatmap(text)

# Layer 6, all heads averaged
visualizer.plot_attention_heatmap(text, layer=6)

# Token influence: which tokens attend most to 'cat'
visualizer.plot_token_influence(text, target_token_idx=2)

# All heads at layer 0
visualizer.plot_head_diversity(text, layer=0)
```

### Feature Attribution (Integrated Gradients)

```python
from src.feature_attribution import GradientAttribution

model = GradientAttribution("distilbert-base-uncased-finetuned-sst-2-english")
model.plot_attributions("The movie was absolutely brilliant.", target_class=1)
```

### LIME Explanation

```python
from src.feature_attribution import LIMETextExplainer

lime = LIMETextExplainer()
lime.plot_explanation("The movie was absolutely brilliant.", target_class=1)
```

### Layer-wise Probing

```python
from src.probing_classifiers import HiddenStateExtractor, ProbingExperiment

extractor = HiddenStateExtractor("bert-base-uncased")
probe = ProbingExperiment(extractor)

texts = ["I loved this movie!", "It was terrible.", ...]
labels = ["pos", "neg", ...]

scores = probe.run(texts, labels, task_name="Sentiment")
probe.plot_layer_probing(scores, task_name="Sentiment")
```

---

## Sample Outputs

- `results/avg_attention.png` — Global average attention heatmap
- `results/layer6_attention.png` — Layer-6 attention heatmap
- `results/token_influence.png` — Influence bar plot for a target token
- `results/gradient_attribution.png` — Integrated gradients attribution
- `results/lime_explanation.png` — LIME token importance
- `results/sentiment_probe.png` — Layer-wise probing accuracy curve

---

## Research Context

This project is part of research on **mechanistic interpretability** — understanding *why* transformer models make specific predictions by analyzing their internal computational structures. Key concepts explored:

- **Attention as routing**: how attention heads selectively focus on task-relevant tokens
- **Representation geometry**: what linguistic structure each layer captures (via probing)
- **Attribution faithfulness**: comparing gradient-based vs. perturbation-based explanations

---

## Author

**Trinadh Kolluboyina**  
AI Engineer & ML Researcher | [LinkedIn](https://linkedin.com/in/trinadhkolluboyina) | [GitHub](https://github.com/trinadh3nadh)
