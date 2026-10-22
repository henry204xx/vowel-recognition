"""Small, plots for available results."""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_confusion(metrics: dict, title: str, path: Path) -> None:
    if not metrics.get("available"): return
    labels, matrix = metrics["labels"], np.array(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(5, 4)); image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
             xlabel="Prediksi", ylabel="Label benar", title=title)
    for i in range(len(labels)):
        for j in range(len(labels)): axis.text(j, i, matrix[i, j], ha="center", va="center")
    fig.colorbar(image); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def save_accuracy_comparison(all_metrics: dict, path: Path) -> None:
    rows = [(system, scenario, metric["accuracy"]) for system, scenarios in all_metrics.items()
            for scenario, metric in scenarios.items() if metric.get("available")]
    if not rows: return
    names = [f"{s}\n{c}" for s, c, _ in rows]; values = [v for _, _, v in rows]
    fig, axis = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4)); axis.bar(names, values); axis.set_ylim(0, 100); axis.set_ylabel("Akurasi (%)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
