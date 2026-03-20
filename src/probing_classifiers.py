"""
Probing Classifiers for Transformer Hidden States
Trains lightweight classifiers on layer representations to understand
what linguistic information each layer encodes.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


class HiddenStateExtractor:
    """Extracts hidden states from all transformer layers."""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def extract(self, texts: List[str], pooling: str = "cls") -> np.ndarray:
        """
        Extract representations from all layers.

        Args:
            texts: List of input sentences
            pooling: 'cls' uses [CLS] token, 'mean' averages all tokens

        Returns:
            Array of shape (num_layers+1, num_texts, hidden_size)
        """
        all_hidden = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=128, padding="max_length"
            )
            with torch.no_grad():
                outputs = self.model(**inputs)

            # hidden_states: tuple of (num_layers+1) tensors, each (1, seq, hidden)
            layers = []
            for hs in outputs.hidden_states:
                if pooling == "cls":
                    rep = hs[0, 0, :].numpy()  # [CLS] token
                else:
                    mask = inputs["attention_mask"][0].unsqueeze(-1).float()
                    rep = (hs[0] * mask).sum(0).numpy() / mask.sum().item()
                layers.append(rep)
            all_hidden.append(layers)

        # Reshape to (num_layers, num_texts, hidden_size)
        num_layers = len(all_hidden[0])
        result = np.array([[all_hidden[i][l] for i in range(len(texts))] for l in range(num_layers)])
        return result


class ProbingExperiment:
    """
    Trains logistic regression probes on each layer's representations
    to measure how much linguistic knowledge is encoded per layer.
    """

    def __init__(self, extractor: HiddenStateExtractor):
        self.extractor = extractor

    def run(
        self,
        texts: List[str],
        labels: List[str],
        task_name: str = "Probing Task",
        cv: int = 5,
        pooling: str = "cls",
    ) -> Dict[int, float]:
        """
        Runs probing experiment across all layers.

        Args:
            texts: Input sentences
            labels: Target labels for the probing task
            task_name: Name used in plot title
            cv: Cross-validation folds
            pooling: Pooling strategy for sentence representation

        Returns:
            Dictionary mapping layer_index -> mean CV accuracy
        """
        print(f"[INFO] Extracting hidden states for {len(texts)} samples...")
        hidden_states = self.extractor.extract(texts, pooling=pooling)

        le = LabelEncoder()
        y = le.fit_transform(labels)

        layer_scores = {}
        for layer_idx in range(hidden_states.shape[0]):
            X = hidden_states[layer_idx]
            clf = LogisticRegression(max_iter=1000, C=1.0)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            layer_scores[layer_idx] = scores.mean()
            print(f"  Layer {layer_idx:2d}: Accuracy = {scores.mean():.4f} ± {scores.std():.4f}")

        return layer_scores

    def plot_layer_probing(
        self,
        layer_scores: Dict[int, float],
        task_name: str = "Probing Task",
        save_path: Optional[str] = None,
    ):
        """Line plot of probing accuracy per layer."""
        layers = sorted(layer_scores.keys())
        accs = [layer_scores[l] for l in layers]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(layers, accs, marker="o", linewidth=2.5, color="#3498db", markersize=8)
        ax.fill_between(layers, accs, alpha=0.15, color="#3498db")
        ax.set_xlabel("Transformer Layer", fontsize=12)
        ax.set_ylabel("Probing Accuracy (CV)", fontsize=12)
        ax.set_title(f"Layer-wise Probing Results — {task_name}", fontsize=13)
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{l}" for l in layers], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved probing plot to {save_path}")
        plt.show()


# ---------- Sample probe tasks ----------

def sentiment_probe_data():
    """Small sentiment dataset for probing."""
    texts = [
        "I loved this movie, it was fantastic!",
        "The film was terrible and boring.",
        "An outstanding performance by the cast.",
        "Worst movie I have ever seen.",
        "Absolutely wonderful and heartwarming.",
        "Dull, lifeless, and predictable.",
        "A masterpiece of modern cinema.",
        "Completely disappointing from start to finish.",
        "The acting was superb and the story was gripping.",
        "I would not recommend this to anyone.",
    ]
    labels = ["pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg", "pos", "neg"]
    return texts, labels


def pos_tag_probe_data():
    """Token-level POS probe (simplified sentence-level proxy)."""
    texts = [
        "The quick brown fox jumps.",     # noun-heavy
        "Running fast and jumping high.",  # verb-heavy
        "Beautiful, wonderful, amazing!",  # adj-heavy
        "She quickly and silently moved.", # adv-heavy
        "The old red wooden chair stood.", # noun-heavy
        "Singing, dancing, and laughing.", # verb-heavy
        "Tiny, fragile, delicate glass.",  # adj-heavy
        "Slowly, carefully, deliberately.",# adv-heavy
    ]
    labels = ["noun", "verb", "adj", "adv", "noun", "verb", "adj", "adv"]
    return texts, labels


if __name__ == "__main__":
    extractor = HiddenStateExtractor("bert-base-uncased")
    probe = ProbingExperiment(extractor)

    print("=== Sentiment Probing ===")
    texts, labels = sentiment_probe_data()
    scores = probe.run(texts, labels, task_name="Sentiment", cv=3)
    probe.plot_layer_probing(scores, task_name="Sentiment", save_path="results/sentiment_probe.png")

    print("\n=== POS Probing ===")
    texts2, labels2 = pos_tag_probe_data()
    scores2 = probe.run(texts2, labels2, task_name="POS (proxy)", cv=3)
    probe.plot_layer_probing(scores2, task_name="POS (proxy)", save_path="results/pos_probe.png")
