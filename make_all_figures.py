"""
Master orchestrator — runs every figure/metric-generating script in
the project in the correct order, end to end. Useful as a single
"reproduce my report" command.

Activate the venv first:  `source venv/bin/activate`
Then run:                 `python make_all_figures.py`
"""


import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TASK1 = [
    "task1_nlp/src/01_explore.py",
    "task1_nlp/src/02_spam_detector.py",
    "task1_nlp/src/03_sentiment.py",
    "task1_nlp/src/04_glove_sentiment.py",
    "task1_nlp/src/05_failure_analysis.py",
    "task1_nlp/src/06_nltk_external.py",
    "task1_nlp/src/07_predict_test.py",
    "task1_nlp/src/08_diagrams.py",
]

TASK2 = [
    "task2_cv/src/01_explore.py",
    "task2_cv/src/02_baseline_mean.py",
    "task2_cv/src/03_hog_ridge.py",
    "task2_cv/src/04_cascaded.py",
    "task2_cv/src/05_evaluate.py",
    "task2_cv/src/06_robustness.py",
    "task2_cv/src/07_predict_test.py",
    "task2_cv/src/08_diagrams.py",
    "task2_cv/src/09_compare_charts.py",
]


def run(script: str) -> None:
    print(f"\n========== {script} ==========")
    res = subprocess.run([sys.executable, str(ROOT / script)],
                         cwd=str(ROOT))
    if res.returncode != 0:
        sys.exit(f"\nFAILED: {script} (exit code {res.returncode})")


def main():
    print("Sussex AML — reproducing every figure/metric ...")
    for s in TASK1 + TASK2:
        run(s)
    print("\nAll figures regenerated under task{1,2}/figures and "
          "all metrics under task{1,2}/outputs.")


if __name__ == "__main__":
    main()
