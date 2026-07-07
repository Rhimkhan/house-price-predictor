"""
src/performance_monitor.py
===========================
Tracks model prediction logs, detects data drift, and generates
a performance dashboard report.

Usage:
    from performance_monitor import PerformanceMonitor
    monitor = PerformanceMonitor()
    monitor.log_prediction(features, predicted_price, actual_price=None)
    monitor.generate_report()
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

ROOT       = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
LOG_FILE   = os.path.join(OUTPUT_DIR, "prediction_log.jsonl")
REPORT_DIR = os.path.join(OUTPUT_DIR, "monitor_reports")
os.makedirs(REPORT_DIR, exist_ok=True)


class PerformanceMonitor:
    """Logs predictions and detects model/data drift over time."""

    def __init__(self):
        self.log_file   = LOG_FILE
        self.report_dir = REPORT_DIR
        self._entries   = self._load_log()

    # ── Logging ───────────────────────────────────────────────

    def log_prediction(self,
                       features: dict,
                       predicted_price: float,
                       actual_price: Optional[float] = None,
                       session_id: str = "") -> str:
        """Append a prediction record to the JSONL log."""
        entry = {
            "id"             : hashlib.md5(
                f"{datetime.now().isoformat()}{predicted_price}".encode()
            ).hexdigest()[:8],
            "timestamp"      : datetime.now().isoformat(),
            "session_id"     : session_id,
            "predicted_price": float(predicted_price),
            "actual_price"   : float(actual_price) if actual_price else None,
            "error"          : float(abs(actual_price - predicted_price)) if actual_price else None,
            "pct_error"      : float(abs(actual_price - predicted_price) / actual_price * 100)
                               if actual_price else None,
            "features"       : {k: float(v) if isinstance(v, (int, float)) else str(v)
                                for k, v in features.items()},
        }
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._entries.append(entry)
        logger.info(f"Logged prediction #{len(self._entries)}: ${predicted_price:,.0f}")
        return entry["id"]

    def _load_log(self) -> list:
        """Load existing log entries."""
        if not os.path.exists(self.log_file):
            return []
        entries = []
        with open(self.log_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    # ── Statistics ────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return summary statistics from the prediction log."""
        if not self._entries:
            return {"total_predictions": 0}

        prices = [e["predicted_price"] for e in self._entries]
        errors = [e["pct_error"] for e in self._entries if e["pct_error"] is not None]

        stats = {
            "total_predictions" : len(self._entries),
            "avg_predicted_price": float(np.mean(prices)),
            "min_predicted_price": float(np.min(prices)),
            "max_predicted_price": float(np.max(prices)),
            "std_predicted_price": float(np.std(prices)),
            "predictions_with_actual": len(errors),
            "mean_pct_error": float(np.mean(errors)) if errors else None,
            "last_prediction": self._entries[-1]["timestamp"],
        }
        return stats

    # ── Drift detection ───────────────────────────────────────

    def detect_drift(self, reference_mean: float = 180_000,
                     reference_std: float = 79_000,
                     threshold: float = 2.0) -> dict:
        """
        Simple z-score drift detection on predicted price distribution.
        Returns drift flag and z-score.
        """
        if len(self._entries) < 10:
            return {"drift_detected": False, "reason": "Not enough data (<10 predictions)"}

        prices  = [e["predicted_price"] for e in self._entries[-50:]]
        current_mean = np.mean(prices)
        z_score = abs(current_mean - reference_mean) / (reference_std / np.sqrt(len(prices)))

        drift = bool(z_score > threshold)
        return {
            "drift_detected"  : drift,
            "z_score"         : float(round(z_score, 3)),
            "current_mean"    : float(round(current_mean, 2)),
            "reference_mean"  : reference_mean,
            "threshold"       : threshold,
            "recommendation"  : "Retrain model" if drift else "Model stable",
        }

    # ── Report ────────────────────────────────────────────────

    def generate_report(self):
        """Generate a visual monitoring report and save to outputs/monitor_reports/."""
        if len(self._entries) < 2:
            logger.warning("Not enough entries to generate a report")
            return

        timestamps = pd.to_datetime([e["timestamp"] for e in self._entries])
        prices     = [e["predicted_price"] for e in self._entries]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Model Performance Monitor", fontsize=14, fontweight="bold")

        # 1. Price over time
        axes[0,0].plot(timestamps, prices, color="#6c63ff", lw=2, marker="o", ms=5)
        axes[0,0].set_title("Predicted Price Over Time")
        axes[0,0].set_ylabel("Price ($)")
        axes[0,0].tick_params(axis="x", rotation=30)
        axes[0,0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

        # 2. Price distribution
        axes[0,1].hist(prices, bins=20, color="#2cb67d", edgecolor="white", alpha=0.8)
        axes[0,1].axvline(np.mean(prices), color="red", ls="--", lw=2,
                          label=f"Mean ${np.mean(prices):,.0f}")
        axes[0,1].set_title("Predicted Price Distribution")
        axes[0,1].legend()
        axes[0,1].xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

        # 3. Error distribution (if actuals exist)
        errors = [e["pct_error"] for e in self._entries if e["pct_error"] is not None]
        if errors:
            axes[1,0].hist(errors, bins=15, color="#f9bc60", edgecolor="white", alpha=0.8)
            axes[1,0].set_title("% Prediction Error Distribution")
            axes[1,0].set_xlabel("% Error")
        else:
            axes[1,0].text(0.5, 0.5, "No actual prices logged\n(% error unavailable)",
                           ha="center", va="center", transform=axes[1,0].transAxes,
                           fontsize=12, color="gray")
            axes[1,0].set_title("% Error Distribution")

        # 4. Prediction count over time (cumulative)
        axes[1,1].plot(timestamps, range(1, len(timestamps)+1),
                       color="#00d2ff", lw=2, drawstyle="steps-post")
        axes[1,1].set_title("Cumulative Predictions")
        axes[1,1].set_ylabel("Count")
        axes[1,1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.report_dir, f"monitor_report_{ts}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Report saved → {path}")
        return path

    def print_dashboard(self):
        """Print a coloured terminal dashboard."""
        G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; B = "\033[1m"; R = "\033[0m"
        stats = self.get_stats()
        drift = self.detect_drift()

        print(f"\n{B}{'═'*55}")
        print(f"  PERFORMANCE MONITOR DASHBOARD")
        print(f"{'═'*55}{R}")
        print(f"  {C}Total Predictions  :{R} {stats['total_predictions']}")
        if stats['total_predictions'] > 0:
            print(f"  {C}Avg Predicted Price:{R} ${stats['avg_predicted_price']:>12,.0f}")
            print(f"  {C}Min / Max          :{R} ${stats['min_predicted_price']:,.0f} / ${stats['max_predicted_price']:,.0f}")
            print(f"  {C}Last Prediction    :{R} {stats['last_prediction']}")
            if stats.get('mean_pct_error') is not None:
                print(f"  {C}Mean % Error       :{R} {stats['mean_pct_error']:.2f}%")
        drift_color = Y if drift['drift_detected'] else G
        print(f"\n  {drift_color}Drift Status       : {'⚠ DRIFT DETECTED' if drift['drift_detected'] else '✓ Model Stable'}{R}")
        print(f"  {C}Recommendation     :{R} {drift.get('recommendation','N/A')}")
        print(f"{B}{'═'*55}{R}\n")
