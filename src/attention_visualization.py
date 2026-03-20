"""
Attention Visualization for Transformer Models
Extracts and visualizes attention weights across layers and heads.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class AttentionExtractor:
    """
    Extracts attention weights from transformer models (BERT, GPT-2, RoBERTa, etc.)
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_attentions=True)
        self.model.eval()
        print(f"[INFO] Loaded model: {model_name}")

    def get_attentions(self, text: str) -> Tuple[List[str], torch.Tensor]:
        """
        Returns tokenized input and attention tensors.

        Args:
            text: Input sentence

        Returns:
            tokens: List of token strings
            attentions: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Stack: (num_layers, 1, num_heads, seq_len, seq_len) -> (num_layers, num_heads, seq_len, seq_len)
        attentions = torch.stack(outputs.attentions).squeeze(1)
        return tokens, attentions

    def get_average_attention(self, attentions: torch.Tensor) -> torch.Tensor:
        """Average attention across all heads and layers."""
        return attentions.mean(dim=(0, 1))  # (seq_len, seq_len)

    def get_layer_attention(self, attentions: torch.Tensor, layer: int) -> torch.Tensor:
        """Average attention across heads for a specific layer."""
        return attentions[layer].mean(dim=0)  # (seq_len, seq_len)


class AttentionVisualizer:
    """
    Visualizes attention patterns using heatmaps and token influence plots.
    """

    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor

    def plot_attention_heatmap(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plots attention heatmap for a given text.

        Args:
            text: Input text
            layer: Specific layer index (None = average across all)
            head: Specific head index (None = average across heads)
            save_path: File path to save the figure
            figsize: Figure dimensions
        """
        tokens, attentions = self.extractor.get_attentions(text)

        if layer is not None and head is not None:
            attn_matrix = attentions[layer][head].numpy()
            title = f"Attention — Layer {layer}, Head {head}"
        elif layer is not None:
            attn_matrix = self.extractor.get_layer_attention(attentions, layer).numpy()
            title = f"Attention — Layer {layer} (avg heads)"
        else:
            attn_matrix = self.extractor.get_average_attention(attentions).numpy()
            title = "Attention — All Layers & Heads (avg)"

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            attn_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            linewidths=0.3,
            linecolor="grey",
        )
        ax.set_title(title, fontsize=14, pad=12)
        ax.set_xlabel("Key Tokens", fontsize=11)
        ax.set_ylabel("Query Tokens", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved heatmap to {save_path}")
        plt.show()

    def plot_token_influence(
        self,
        text: str,
        target_token_idx: int,
        layer: Optional[int] = None,
        save_path: Optional[str] = None,
    ):
        """
        Bar plot showing how much each token attends to the target token.

        Args:
            text: Input text
            target_token_idx: Index of the token to analyze influence on
            layer: Layer to analyze (None = all layers averaged)
            save_path: File path to save the figure
        """
        tokens, attentions = self.extractor.get_attentions(text)

        if layer is not None:
            attn_matrix = self.extractor.get_layer_attention(attentions, layer).numpy()
            layer_label = f"Layer {layer}"
        else:
            attn_matrix = self.extractor.get_average_attention(attentions).numpy()
            layer_label = "All Layers (avg)"

        target_token = tokens[target_token_idx]
        influence = attn_matrix[:, target_token_idx]

        colors = ["#e74c3c" if i == target_token_idx else "#3498db" for i in range(len(tokens))]

        fig, ax = plt.subplots(figsize=(12, 4))
        bars = ax.bar(tokens, influence, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_title(
            f"Token Influence on '{target_token}' — {layer_label}",
            fontsize=13, pad=10
        )
        ax.set_xlabel("Tokens", fontsize=11)
        ax.set_ylabel("Attention Weight", fontsize=11)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved token influence plot to {save_path}")
        plt.show()

    def plot_head_diversity(
        self,
        text: str,
        layer: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Plots each attention head separately for a given layer.

        Args:
            text: Input text
            layer: Layer to visualize
            save_path: File path to save the figure
        """
        tokens, attentions = self.extractor.get_attentions(text)
        num_heads = attentions.shape[1]
        cols = 4
        rows = (num_heads + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            attn_matrix = attentions[layer][head_idx].numpy()
            sns.heatmap(
                attn_matrix,
                xticklabels=tokens if len(tokens) < 15 else False,
                yticklabels=tokens if len(tokens) < 15 else False,
                cmap="magma",
                ax=axes[head_idx],
                cbar=False,
            )
            axes[head_idx].set_title(f"Head {head_idx}", fontsize=9)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"All Attention Heads — Layer {layer}", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Saved head diversity plot to {save_path}")
        plt.show()


if __name__ == "__main__":
    sample_text = "The cat sat on the mat near the warm fireplace."

    extractor = AttentionExtractor("bert-base-uncased")
    visualizer = AttentionVisualizer(extractor)

    print("\n[1] Average attention heatmap:")
    visualizer.plot_attention_heatmap(sample_text, save_path="results/avg_attention.png")

    print("\n[2] Layer 6 attention heatmap:")
    visualizer.plot_attention_heatmap(sample_text, layer=6, save_path="results/layer6_attention.png")

    print("\n[3] Token influence on 'cat' (index 2):")
    visualizer.plot_token_influence(sample_text, target_token_idx=2, save_path="results/token_influence.png")

    print("\n[4] Head diversity for Layer 0:")
    visualizer.plot_head_diversity(sample_text, layer=0, save_path="results/head_diversity.png")
