"""
utils/evaluator.py
------------------
Everything needed to measure and visualize model performance.

METRICS EXPLAINED (for beginners)
----------------------------------
Accuracy  = (correct predictions) / (total predictions)
            → Simple but misleading if classes are imbalanced.

Precision = (true positives) / (true positives + false positives)
            → "Of everything the model labelled as 'sports', how much WAS sports?"

Recall    = (true positives) / (true positives + false negatives)
            → "Of all real 'sports' articles, how many did we catch?"

F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)
            → Harmonic mean — penalises models that are good at one but bad at the other.

Confusion Matrix — rows = actual class, columns = predicted class.
   - Perfect model: high values only on the diagonal.
   - Off-diagonal values show which classes the model CONFUSES with each other.
"""

import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:

    def __init__(self, class_names=None):
        self.class_names = class_names

    # ─── Core evaluation ──────────────────────────────────────────────────────

    def evaluate(self, y_true, y_pred, y_proba=None):
        """
        Compute and print all metrics.

        Parameters
        ----------
        y_true  : array-like   true labels
        y_pred  : array-like   predicted labels
        y_proba : array-like   predicted probabilities (optional, for AUC)

        Returns
        -------
        dict with all metric values
        """
        acc = accuracy_score(y_true, y_pred)

        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

        print("\n" + "=" * 70)
        print("  MODEL EVALUATION RESULTS")
        print("=" * 70)
        print(f"\n  Overall Accuracy : {acc:.4f}  ({acc*100:.1f}%)")

        # Macro averages
        macro = report.get("macro avg", {})
        print(f"  Macro Precision  : {macro.get('precision', 0):.4f}")
        print(f"  Macro Recall     : {macro.get('recall', 0):.4f}")
        print(f"  Macro F1-Score   : {macro.get('f1-score', 0):.4f}")

        # ROC-AUC (only if probabilities provided)
        auc = None
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                print(f"  Macro ROC-AUC    : {auc:.4f}")
            except Exception:
                pass

        print("\n  Per-class breakdown:")
        print("-" * 70)
        print(f"  {'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print("-" * 70)

        names = self.class_names or [k for k in report if k not in
                                     ("accuracy", "macro avg", "weighted avg")]
        for name in names:
            if name in report:
                r = report[name]
                print(f"  {name:<35} {r['precision']:>10.3f} {r['recall']:>10.3f}"
                      f" {r['f1-score']:>10.3f} {int(r['support']):>10}")
        print("=" * 70)

        return {
            "accuracy": acc,
            "macro_precision": macro.get("precision", 0),
            "macro_recall":    macro.get("recall", 0),
            "macro_f1":        macro.get("f1-score", 0),
            "roc_auc":         auc,
            "report":          report,
        }

    # ─── Confusion matrix ─────────────────────────────────────────────────────

    def plot_confusion_matrix(self, y_true, y_pred, save_path="confusion_matrix.png"):
        """
        Plot and save a heatmap confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred)

        # Normalise to percentages so all classes are comparable
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle("Confusion Matrix", fontsize=16, fontweight="bold", y=1.01)

        for ax, data, title, fmt in [
            (axes[0], cm,      "Counts",      "d"),
            (axes[1], cm_norm, "Normalised",  ".2f"),
        ]:
            sns.heatmap(
                data,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=self.class_names or "auto",
                yticklabels=self.class_names or "auto",
                ax=ax,
                linewidths=0.5,
            )
            ax.set_title(title, fontsize=13)
            ax.set_xlabel("Predicted Label", fontsize=11)
            ax.set_ylabel("True Label",      fontsize=11)
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            ax.tick_params(axis="y", rotation=0,  labelsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[EVAL] Confusion matrix saved → {save_path}")
        return cm

    # ─── Training curves ──────────────────────────────────────────────────────

    def plot_training_sizes(self, clf_factory, X, y, save_path="learning_curve.png"):
        """
        Plot accuracy vs. training set size (learning curve).
        Helps diagnose underfitting / overfitting.
        """
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, val_scores = learning_curve(
            clf_factory(), X, y,
            cv=5, scoring="accuracy",
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1,
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train accuracy",  color="#2196F3")
        ax.plot(train_sizes, val_scores.mean(axis=1),   "s-", label="Val accuracy",    color="#FF5722")
        ax.fill_between(train_sizes,
                        train_scores.mean(1) - train_scores.std(1),
                        train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#2196F3")
        ax.fill_between(train_sizes,
                        val_scores.mean(1)   - val_scores.std(1),
                        val_scores.mean(1)   + val_scores.std(1),   alpha=0.15, color="#FF5722")
        ax.set_xlabel("Training set size")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve — more data = better model?")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[EVAL] Learning curve saved → {save_path}")

    # ─── Model comparison bar chart ───────────────────────────────────────────

    @staticmethod
    def plot_model_comparison(results: dict, save_path="model_comparison.png"):
        """
        results = {"LogisticRegression": 0.92, "SVM": 0.94, ...}
        """
        names  = list(results.keys())
        scores = list(results.values())
        colors = ["#4CAF50" if s == max(scores) else "#90CAF9" for s in scores]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.5)
        ax.set_xlim(max(0, min(scores) - 0.05), min(1.0, max(scores) + 0.05))
        ax.set_xlabel("F1-Score (macro)")
        ax.set_title("Model Comparison")
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{score:.4f}", va="center", fontsize=10)
        ax.axvline(max(scores), color="green", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[EVAL] Model comparison saved → {save_path}")
