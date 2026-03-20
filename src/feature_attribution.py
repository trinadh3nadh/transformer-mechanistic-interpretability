"""
Feature Attribution for Transformer Models
Supports SHAP, LIME, and gradient-based attribution methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class GradientAttribution:
    """
    Gradient-based feature attribution (Integrated Gradients approximation).
    Works with any HuggingFace sequence classification model.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        print(f"[INFO] Loaded classification model: {model_name}")

    def get_token_attributions(
        self, text: str, target_class: int = 1, n_steps: int = 50
    ) -> Tuple[List[str], np.ndarray]:
        """
        Computes Integrated Gradients attribution scores for each token.

        Args:
            text: Input sentence
            target_class: Class index to attribute (0=negative, 1=positive for SST-2)
            n_steps: Number of integration steps

        Returns:
            tokens: List of token strings
            attributions: Attribution score per token (L2 norm of embedding gradients)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        embeddings = self.model.get_input_embeddings()(inputs["input_ids"])  # (1, seq, hidden)
        baseline = torch.zeros_like(embeddings)

        total_grads = torch.zeros_like(embeddings)
        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline + alpha * (embeddings - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            outputs = self.model(inputs_embeds=interpolated, attention_mask=inputs["attention_mask"])
            score = outputs.logits[0, target_class]
            score.backward()

            total_grads += interpolated.grad.detach()

        # Integrated gradients
        avg_grads = total_grads / n_steps
        integrated_grads = (embeddings - baseline).detach() * avg_grads  # (1, seq, hidden)

        # L2 norm per token
        attributions = integrated_grads.squeeze(0).norm(dim=-1).numpy()
        return tokens, attributions

    def plot_attributions(
        self,
        text: str,
        target_class: int = 1,
        save_path: Optional[str] = None,
    ):
        """Bar plot of token-level attribution scores."""
        tokens, attributions = self.get_token_attributions(text, target_class)
        norm_attr = attributions / (attributions.max() + 1e-8)

        cmap = plt.cm.RdYlGn
        colors = [cmap(v) for v in norm_attr]

        fig, ax = plt.subplots(figsize=(max(10, len(tokens) * 0.7), 4))
        bars = ax.bar(tokens, norm_attr, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(
            f"Integrated Gradients Attribution — Class {target_class}\n\"{text[:60]}...\"" if len(text) > 60 else f"Integrated Gradients Attribution — Class {target_class}",
            fontsize=12,
        )
        ax.set_ylabel("Normalized Attribution Score", fontsize=10)
        ax.set_xlabel("Tokens", fontsize=10)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved attribution plot to {save_path}")
        plt.show()
        return tokens, attributions


class LIMETextExplainer:
    """
    LIME-based local explanation for transformer text classifiers.
    Uses perturbation of token presence to estimate importance.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.classifier = pipeline("text-classification", model=model_name, return_all_scores=True)
        self.label_map = {0: "NEGATIVE", 1: "POSITIVE"}
        print(f"[INFO] Loaded LIME pipeline: {model_name}")

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Returns probability matrix for a list of texts."""
        results = self.classifier(texts, truncation=True, max_length=512)
        probs = []
        for r in results:
            score_map = {item["label"]: item["score"] for item in r}
            probs.append([
                score_map.get("NEGATIVE", 0.0),
                score_map.get("POSITIVE", 0.0),
            ])
        return np.array(probs)

    def explain(
        self,
        text: str,
        num_samples: int = 500,
        target_class: int = 1,
    ) -> Dict[str, float]:
        """
        LIME explanation by token masking perturbations.

        Args:
            text: Input text
            num_samples: Number of random perturbations
            target_class: Class to explain

        Returns:
            Dictionary mapping token -> importance score
        """
        words = text.split()
        n = len(words)
        if n == 0:
            return {}

        # Generate random binary masks (1=keep, 0=mask)
        masks = np.random.randint(0, 2, size=(num_samples, n))
        masks[0] = np.ones(n)  # include original

        perturbed_texts = []
        for mask in masks:
            masked = [w if mask[i] else "[MASK]" for i, w in enumerate(words)]
            perturbed_texts.append(" ".join(masked))

        probs = self._predict_proba(perturbed_texts)
        labels = probs[:, target_class]

        # Weighted least squares regression
        distances = np.sum(masks == 0, axis=1)
        weights = np.exp(-distances / (n * 0.25))

        X = masks.astype(float)
        W = np.diag(weights)
        try:
            coefs = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ labels, rcond=None)[0]
        except np.linalg.LinAlgError:
            coefs = np.zeros(n)

        importance = {words[i]: float(coefs[i]) for i in range(n)}
        return importance

    def plot_explanation(
        self,
        text: str,
        num_samples: int = 500,
        target_class: int = 1,
        top_k: int = 15,
        save_path: Optional[str] = None,
    ):
        """Horizontal bar chart of LIME token importances."""
        importance = self.explain(text, num_samples, target_class)
        sorted_items = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        words, scores = zip(*sorted_items)

        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in scores]

        fig, ax = plt.subplots(figsize=(8, max(4, len(words) * 0.45)))
        y_pos = np.arange(len(words))
        ax.barh(y_pos, scores, color=colors, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("LIME Importance Score", fontsize=11)
        ax.set_title(
            f"LIME Explanation — Class {self.label_map.get(target_class, target_class)} (Top {top_k} tokens)",
            fontsize=12,
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved LIME explanation to {save_path}")
        plt.show()
        return importance


if __name__ == "__main__":
    sample = "The movie was absolutely brilliant and deeply moving."

    print("=== Gradient Attribution ===")
    grad_attr = GradientAttribution()
    grad_attr.plot_attributions(sample, target_class=1, save_path="results/gradient_attribution.png")

    print("\n=== LIME Explanation ===")
    lime_explainer = LIMETextExplainer()
    lime_explainer.plot_explanation(sample, target_class=1, save_path="results/lime_explanation.png")
