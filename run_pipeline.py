#!/usr/bin/env python3
"""
🚀 HOUSE PRICE PREDICTION — COMPLETE AUTOMATED PIPELINE v2.0
=============================================================
Professional portfolio-grade pipeline with coloured output,
multi-mode execution, dependency checking, and progress tracking.

Usage:
    python run_pipeline.py --full        # Run entire pipeline
    python run_pipeline.py --quick       # Predict (requires trained model)
    python run_pipeline.py --web         # Start Flask web app
    python run_pipeline.py --test        # Run test suite
    python run_pipeline.py --monitor     # Show performance dashboard
    python run_pipeline.py --explain     # Generate SHAP explanations
"""

import os
import sys
import subprocess
import argparse

# Fix UnicodeEncodeError on Windows terminals (cp1252 → utf-8)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import shutil
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Terminal colours ────────────────────────────────────────
class C:
    H  = "\033[95m"   # header / magenta
    B  = "\033[94m"   # blue
    G  = "\033[92m"   # green
    Y  = "\033[93m"   # yellow
    R  = "\033[91m"   # red
    C  = "\033[96m"   # cyan
    W  = "\033[97m"   # white
    BD = "\033[1m"    # bold
    DM = "\033[2m"    # dim
    E  = "\033[0m"    # end/reset


class HousePricePipeline:
    """Orchestrates the full ML pipeline with styled console output."""

    def __init__(self):
        self.start_time = datetime.now()
        self._step = 0

    # ── Printing helpers ──────────────────────────────────

    def header(self, text: str):
        print(f"\n{C.BD}{C.H}{'═'*60}")
        print(f"  🚀  {text}")
        print(f"{'═'*60}{C.E}")

    def step(self, text: str):
        self._step += 1
        print(f"\n{C.BD}{C.B}[{self._step}]{C.E} {C.W}{text}{C.E}")
        print(f"    {C.DM}{'─'*50}{C.E}")

    def ok(self,   text): print(f"    {C.G}✅  {text}{C.E}")
    def warn(self, text): print(f"    {C.Y}⚠️  {text}{C.E}")
    def err(self,  text): print(f"    {C.R}❌  {text}{C.E}")
    def info(self, text): print(f"    {C.C}📊  {text}{C.E}")

    def progress(self, label: str, done: int, total: int):
        bar_len = 30
        filled  = int(bar_len * done / total)
        bar     = "█" * filled + "░" * (bar_len - filled)
        pct     = done / total * 100
        print(f"\r    {label}: [{C.G}{bar}{C.E}] {pct:5.1f}%", end="", flush=True)
        if done == total:
            print()

    # ── Checks ────────────────────────────────────────────

    def check_python_version(self):
        self.step("Python Version Check")
        v = sys.version_info
        if v < (3, 8):
            self.err(f"Python 3.8+ required, got {v.major}.{v.minor}")
            sys.exit(1)
        self.ok(f"Python {v.major}.{v.minor}.{v.micro}")

    def check_dependencies(self) -> bool:
        self.step("Dependency Check")
        required = [
            ("pandas",      "pandas"),
            ("numpy",       "numpy"),
            ("matplotlib",  "matplotlib"),
            ("seaborn",     "seaborn"),
            ("sklearn",     "scikit-learn"),
            ("xgboost",     "xgboost"),
            ("lightgbm",    "lightgbm"),
            ("catboost",    "catboost"),
            ("joblib",      "joblib"),
            ("flask",       "flask"),
            ("yaml",        "pyyaml"),
        ]
        missing_pip = []
        for import_name, pip_name in required:
            try:
                __import__(import_name)
            except ImportError:
                missing_pip.append(pip_name)

        if missing_pip:
            self.warn(f"Missing packages: {', '.join(missing_pip)}")
            self.info("Auto-installing…")
            ret = subprocess.call(
                [sys.executable, "-m", "pip", "install"] + missing_pip + ["-q"])
            if ret == 0:
                self.ok("All packages installed")
            else:
                self.err("Auto-install failed — run: pip install -r requirements.txt")
                return False
        else:
            self.ok(f"All {len(required)} packages satisfied")
        return True

    def check_data(self) -> bool:
        self.step("Data Validation")
        train = os.path.join(ROOT, "data", "train.csv")
        test  = os.path.join(ROOT, "data", "test.csv")
        ok    = True

        for path, name in [(train, "train.csv"), (test, "test.csv")]:
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                self.ok(f"{name}  ({size:,.0f} KB)")
            else:
                self.err(f"{name} NOT FOUND in data/")
                ok = False

        if not ok:
            print(f"""
{C.Y}  Download the dataset from Kaggle:
  https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
  Then place train.csv and test.csv in the data/ folder.{C.E}
""")
        return ok

    def check_model(self) -> bool:
        model_path = os.path.join(ROOT, "src", "model.pkl")
        return os.path.exists(model_path)

    def verify_outputs(self):
        self.step("Verifying Outputs")
        expected = [
            "src/model.pkl", "src/scaler.pkl", "src/label_encoders.pkl",
            "outputs/submission.csv",
            "outputs/eda_plots/missing_values.png",
            "outputs/eda_plots/target_distribution.png",
            "outputs/eda_plots/top_correlations.png",
            "outputs/eda_plots/categorical_analysis.png",
            "outputs/eda_plots/outliers.png",
            "outputs/model_results/model_comparison.png",
            "outputs/model_results/feature_importance.png",
            "outputs/model_results/actual_vs_predicted.png",
            "outputs/model_results/residual_plot.png",
        ]
        found = missing = 0
        for rel in expected:
            path = os.path.join(ROOT, rel.replace("/", os.sep))
            if os.path.exists(path):
                size = os.path.getsize(path) / 1024
                self.ok(f"{rel}  ({size:,.0f} KB)")
                found += 1
            else:
                self.warn(f"{rel}  — not found")
                missing += 1
        self.info(f"Summary: {found} found, {missing} missing")

    # ── Pipeline steps ────────────────────────────────────

    def run_script(self, label: str, script: str, *args) -> bool:
        self.step(label)
        cmd = [sys.executable, os.path.join(ROOT, script)] + list(args)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode == 0:
            self.ok(f"{label} completed")
            return True
        else:
            self.err(f"{label} failed (exit code {result.returncode})")
            return False

    # ── Modes ─────────────────────────────────────────────

    def run_full(self):
        self.header("HOUSE PRICE PREDICTION — FULL PIPELINE")
        self.info(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.check_python_version()
        if not self.check_dependencies():
            return
        if not self.check_data():
            return

        if not self.run_script("Model Training (all 9 models + tuning)",
                               "src/model_training.py"):
            return

        self.verify_outputs()

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.header(f"✨  PIPELINE COMPLETE  ({elapsed:.0f}s)")
        print(f"""
  {C.G}{C.BD}Next steps:{C.E}
    {C.C}→{C.E}  Predict interactively : {C.BD}python run_pipeline.py --quick{C.E}
    {C.C}→{C.E}  Start web app         : {C.BD}python run_pipeline.py --web{C.E}
    {C.C}→{C.E}  Run tests             : {C.BD}python run_pipeline.py --test{C.E}
    {C.C}→{C.E}  Submit to Kaggle      : Upload {C.BD}outputs/submission.csv{C.E}
""")

    def run_quick(self):
        self.header("QUICK INTERACTIVE PREDICTOR")
        if not self.check_model():
            self.err("Model not found — run: python run_pipeline.py --full")
            return
        subprocess.run([sys.executable, os.path.join(ROOT, "src", "predict.py")], cwd=ROOT)

    def run_web(self):
        self.header("FLASK WEB APPLICATION")
        self.info("Starting on http://localhost:5000")
        self.info("Press Ctrl+C to stop")
        try:
            subprocess.run([sys.executable, os.path.join(ROOT, "app.py")], cwd=ROOT)
        except KeyboardInterrupt:
            self.warn("Web server stopped")

    def run_tests(self):
        self.header("TEST SUITE")
        # Install pytest if needed
        try:
            import pytest  # noqa
        except ImportError:
            subprocess.call([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov", "-q"])
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=ROOT
        )
        if result.returncode == 0:
            self.ok("All tests passed")
        else:
            self.warn("Some tests failed — check output above")

    def run_monitor(self):
        self.header("PERFORMANCE MONITOR")
        sys.path.insert(0, os.path.join(ROOT, "src"))
        try:
            from performance_monitor import PerformanceMonitor
            monitor = PerformanceMonitor()
            monitor.print_dashboard()
            report = monitor.generate_report()
            if report:
                self.ok(f"Report saved → {report}")
        except Exception as e:
            self.err(f"Monitor failed: {e}")

    def run_explain(self):
        self.header("SHAP MODEL EXPLAINABILITY")
        if not self.check_model():
            self.err("Model not found — run: python run_pipeline.py --full")
            return
        import joblib, numpy as np
        sys.path.insert(0, os.path.join(ROOT, "src"))
        try:
            model  = joblib.load(os.path.join(ROOT, "src", "model.pkl"))
            scaler = joblib.load(os.path.join(ROOT, "src", "scaler.pkl"))
            from model_explainer import ModelExplainer
            n = scaler.n_features_in_
            X_dummy = np.zeros((50, n))
            feature_names = [f"feat_{i}" for i in range(n)]
            explainer = ModelExplainer(model, X_dummy, feature_names)
            paths = explainer.generate_all_plots(X_dummy[:20])
            for p in paths:
                if p:
                    self.ok(f"Saved → {p}")
        except Exception as e:
            self.err(f"Explanation failed: {e}")

    def show_help(self):
        print(f"""
{C.BD}{C.H}
╔══════════════════════════════════════════════════════════════╗
║  🏠  HOUSE PRICE PREDICTION SYSTEM  v2.0                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  {C.C}python run_pipeline.py --full{C.H}     Run entire pipeline      ║
║  {C.C}python run_pipeline.py --quick{C.H}    Quick interactive predict ║
║  {C.C}python run_pipeline.py --web{C.H}      Start Flask web app       ║
║  {C.C}python run_pipeline.py --test{C.H}     Run test suite            ║
║  {C.C}python run_pipeline.py --monitor{C.H}  Performance dashboard      ║
║  {C.C}python run_pipeline.py --explain{C.H}  Generate SHAP plots        ║
║                                                              ║
║  {C.G}📊  Results   → outputs/{C.H}                              ║
║  {C.G}🤖  Model     → src/model.pkl{C.H}                          ║
║  {C.G}🌐  Web App   → http://localhost:5000{C.H}                  ║
╚══════════════════════════════════════════════════════════════╝
{C.E}""")


def main():
    parser = argparse.ArgumentParser(description="House Price Prediction Pipeline v2.0")
    parser.add_argument("--full",     action="store_true", help="Run entire pipeline")
    parser.add_argument("--quick",    action="store_true", help="Interactive predictor")
    parser.add_argument("--web",      action="store_true", help="Start Flask web app")
    parser.add_argument("--test",     action="store_true", help="Run test suite")
    parser.add_argument("--monitor",  action="store_true", help="Performance dashboard")
    parser.add_argument("--explain",  action="store_true", help="SHAP explanations")
    args = parser.parse_args()

    pipeline = HousePricePipeline()

    if   args.full:    pipeline.run_full()
    elif args.quick:   pipeline.run_quick()
    elif args.web:     pipeline.run_web()
    elif args.test:    pipeline.run_tests()
    elif args.monitor: pipeline.run_monitor()
    elif args.explain: pipeline.run_explain()
    else:              pipeline.show_help()


if __name__ == "__main__":
    main()
