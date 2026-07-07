"""
src/model_explainer.py
======================
SHAP-based model explainability — generates waterfall, summary, and bar plots.

Usage:
    from model_explainer import ModelExplainer
    explainer = ModelExplainer(model, X_train, feature_names)
    explainer.generate_all_plots(X_sample)
"""

import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT      = os.path.join(os.path.dirname(__file__), "..")
SHAP_DIR  = os.path.join(ROOT, "outputs", "shap_plots")
os.makedirs(SHAP_DIR, exist_ok=True)


class ModelExplainer:
    """Wrapper for SHAP-based model explanation."""

    def __init__(self, model, X_background: np.ndarray, feature_names: list):
        self.model         = model
        self.X_background  = X_background
        self.feature_names = feature_names
        self._explainer    = None
        self._shap_values  = None

    def _build_explainer(self):
        """Lazy-initialise SHAP explainer."""
        try:
            import shap
            # TreeExplainer for gradient boosted trees (fast)
            self._explainer = shap.TreeExplainer(
                self.model,
                data=self.X_background[:100],
                feature_perturbation="interventional",
            )
            logger.info("SHAP TreeExplainer initialised")
        except ImportError:
            logger.warning("shap not installed — run: pip install shap")
            raise
        except Exception:
            import shap
            self._explainer = shap.Explainer(self.model, self.X_background[:100])

    def compute_shap_values(self, X: np.ndarray):
        """Compute and cache SHAP values for X."""
        if self._explainer is None:
            self._build_explainer()
        import shap
        self._shap_values = self._explainer(X)
        logger.info(f"SHAP values computed for {X.shape[0]} samples")
        return self._shap_values

    # ── Plots ──────────────────────────────────────────────────

    def plot_summary(self, X: np.ndarray, max_display: int = 20):
        """Beeswarm summary plot showing feature impact distribution."""
        try:
            import shap
            sv = self.compute_shap_values(X)
            shap.summary_plot(sv, X,
                              feature_names=self.feature_names,
                              max_display=max_display,
                              show=False)
            path = os.path.join(SHAP_DIR, "shap_summary.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved → {path}")
            return path
        except Exception as e:
            logger.error(f"SHAP summary failed: {e}")

    def plot_bar_importance(self, X: np.ndarray, max_display: int = 20):
        """Bar chart of mean absolute SHAP values."""
        try:
            import shap
            sv = self.compute_shap_values(X)
            shap.summary_plot(sv, X,
                              feature_names=self.feature_names,
                              plot_type="bar",
                              max_display=max_display,
                              show=False)
            path = os.path.join(SHAP_DIR, "shap_bar_importance.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved → {path}")
            return path
        except Exception as e:
            logger.error(f"SHAP bar plot failed: {e}")

    def plot_waterfall(self, X: np.ndarray, sample_idx: int = 0):
        """Waterfall plot for a single prediction."""
        try:
            import shap
            sv = self.compute_shap_values(X)
            shap.waterfall_plot(sv[sample_idx], show=False)
            path = os.path.join(SHAP_DIR, f"shap_waterfall_sample{sample_idx}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved → {path}")
            return path
        except Exception as e:
            logger.error(f"SHAP waterfall failed: {e}")

    def manual_importance_plot(self, X: np.ndarray):
        """
        Fallback feature importance bar chart using model's built-in
        feature_importances_ — used when SHAP is unavailable.
        """
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Model has no feature_importances_")
            return

        imp = self.model.feature_importances_
        idx = np.argsort(imp)[::-1][:20]

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(idx)))
        ax.barh([self.feature_names[i] for i in reversed(idx)],
                imp[list(reversed(idx))],
                color=list(reversed(colors)), edgecolor="white")
        ax.set_title("Top-20 Feature Importances (Built-in)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance Score")
        plt.tight_layout()
        path = os.path.join(SHAP_DIR, "feature_importance_builtin.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved → {path}")
        return path

    def generate_all_plots(self, X: np.ndarray):
        """Generate all available explanation plots."""
        logger.info("Generating all SHAP/importance plots…")
        paths = []
        try:
            import shap  # noqa: F401
            paths.append(self.plot_summary(X))
            paths.append(self.plot_bar_importance(X))
            paths.append(self.plot_waterfall(X, sample_idx=0))
        except ImportError:
            logger.info("SHAP not installed — generating built-in importance only")
            paths.append(self.manual_importance_plot(X))
        return [p for p in paths if p]


def explain_single_prediction(model, scaler, feature_names: list, features_dict: dict) -> dict:
    """
    Return top-5 feature contributions for a single prediction (heuristic).
    Works without SHAP by using the model's feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return {}

    imp = model.feature_importances_
    ranked = sorted(
        zip(feature_names, imp),
        key=lambda x: x[1], reverse=True
    )[:5]

    return {name: float(score) for name, score in ranked}
