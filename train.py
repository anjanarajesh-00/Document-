"""
train.py
--------
THE MAIN TRAINING SCRIPT — run this to build your classifier.

Usage:
    python train.py                          # train with defaults
    python train.py --model svm             # choose a specific model
    python train.py --compare               # compare ALL models
    python train.py --feature bert          # use BERT embeddings

What happens here (step by step):
1.  Load data (20 Newsgroups demo, or your own via data/sample_data.py)
2.  Split into train / test sets (80% / 20%)
3.  Preprocess text (clean, tokenize, lemmatize)
4.  Extract features (TF-IDF by default)
5.  Handle class imbalance (optional, via class_weight="balanced")
6.  Train model(s)
7.  Evaluate with all metrics
8.  Save best model to disk
9.  Plot confusion matrix + model comparison
"""

import argparse
import logging
import os
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

# ── Add project root to path so imports work from any directory ──────────────
sys.path.insert(0, os.path.dirname(__file__))

from data.sample_data        import load_dataset
from utils.preprocessor      import TextPreprocessor
from utils.feature_extractor import TFIDFExtractor, get_extractor
from utils.evaluator         import ModelEvaluator

# ── Classifiers ───────────────────────────────────────────────────────────────
from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes     import MultinomialNB
from sklearn.calibration     import CalibratedClassifierCV

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Model registry ───────────────────────────────────────────────────────────
# WHY THESE MODELS?
# ------------------
# LogisticRegression : fast, interpretable, great baseline. Works well on TF-IDF.
# LinearSVC          : best classic model for high-dim text. Very fast, accurate.
# MultinomialNB      : works with raw TF-IDF counts. Extremely fast, low memory.
# RandomForest       : good for feature importance analysis, slower.
# GradientBoosting   : often best classic ML accuracy but slow to train.

MODELS = {
    "logistic": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # handles class imbalance automatically
        C=5.0,                     # regularisation strength (tune if needed)
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=-1,
    ),
    "svm": CalibratedClassifierCV(   # wrap LinearSVC to get predict_proba()
        LinearSVC(
            class_weight="balanced",
            C=1.0,
            max_iter=2000,
        )
    ),
    "naive_bayes": MultinomialNB(alpha=0.1),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    ),
}


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    model_key="svm",
    feature_method="tfidf",
    compare_all=False,
    output_dir="models",
):
    os.makedirs(output_dir, exist_ok=True)

    # ── STEP 1: Load data ─────────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STEP 1 — Load Dataset")
    print("━" * 70)
    texts, labels = load_dataset()
    class_names   = sorted(set(labels))
    num_classes   = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")

    # ── STEP 2: Train / test split ────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STEP 2 — Train / Test Split  (80% / 20%)")
    print("━" * 70)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,   # ensure balanced split across classes
    )
    print(f"  Train: {len(X_train_raw):>6,} documents")
    print(f"  Test : {len(X_test_raw):>6,} documents")

    # ── STEP 3: Preprocess ────────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STEP 3 — Text Preprocessing")
    print("━" * 70)
    preprocessor = TextPreprocessor(method="lemmatize")

    # Show a before/after example
    print("\n  [BEFORE]:", X_train_raw[0][:200])
    cleaned_example = preprocessor.clean(X_train_raw[0])
    print("\n  [AFTER] :", cleaned_example[:200])
    print()

    print("  Cleaning training texts...")
    X_train_clean = preprocessor.clean_batch(X_train_raw)
    print("  Cleaning test texts...")
    X_test_clean  = preprocessor.clean_batch(X_test_raw)

    # ── STEP 4: Feature extraction ────────────────────────────────────────────
    print("\n" + "━" * 70)
    print(f"  STEP 4 — Feature Extraction  [{feature_method.upper()}]")
    print("━" * 70)
    extractor = get_extractor(feature_method)
    X_train   = extractor.fit_transform(X_train_clean)
    X_test    = extractor.transform(X_test_clean)
    print(f"  Feature matrix shape: {X_train.shape}")

    # ── STEP 5: Train ─────────────────────────────────────────────────────────
    evaluator = ModelEvaluator(class_names=class_names)

    if compare_all:
        _compare_all_models(X_train, X_test, y_train, y_test, evaluator, output_dir)
        _save_pipeline(MODELS["svm"], extractor, preprocessor, class_names, output_dir)
        return

    print("\n" + "━" * 70)
    print(f"  STEP 5 — Training Model  [{model_key.upper()}]")
    print("━" * 70)

    if model_key == "naive_bayes":
        # MultinomialNB requires non-negative values — use raw TF-IDF (sublinear=False)
        from sklearn.feature_extraction.text import TfidfVectorizer
        nb_vec = TfidfVectorizer(max_features=50_000, sublinear_tf=False, min_df=2)
        X_train_nb = nb_vec.fit_transform(X_train_clean)
        X_test_nb  = nb_vec.transform(X_test_clean)
        clf = MODELS[model_key]
        clf.fit(X_train_nb, y_train)
        y_pred = clf.predict(X_test_nb)
        try:
            y_proba = clf.predict_proba(X_test_nb)
        except Exception:
            y_proba = None
    else:
        clf = MODELS[model_key]
        print("  Fitting model (this may take 10–60 seconds)...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            y_proba = clf.predict_proba(X_test)
        except Exception:
            y_proba = None

    # ── 5b: 5-fold cross-validation on training set ───────────────────────────
    print("\n  Running 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"  CV F1 scores  : {cv_scores}")
    print(f"  CV F1 mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── STEP 6: Evaluate ──────────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STEP 6 — Evaluation")
    print("━" * 70)
    metrics = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path=f"{output_dir}/confusion_matrix.png")

    # Most informative features (TF-IDF only)
    if feature_method == "tfidf" and hasattr(clf, "coef_"):
        extractor.top_features_per_class(clf, class_names)

    # ── STEP 7: Save pipeline ─────────────────────────────────────────────────
    _save_pipeline(clf, extractor, preprocessor, class_names, output_dir)

    print("\n" + "━" * 70)
    print("  DONE ✓")
    print("━" * 70)
    print(f"  Final test accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1            : {metrics['macro_f1']:.4f}")
    print(f"  Artefacts saved in  : {output_dir}/")
    print()


# ─── Compare all models ───────────────────────────────────────────────────────

def _compare_all_models(X_train, X_test, y_train, y_test, evaluator, output_dir):
    print("\n" + "━" * 70)
    print("  COMPARING ALL MODELS")
    print("━" * 70)

    results = {}
    for name, clf in MODELS.items():
        print(f"\n  Training {name}...")
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            results[name] = f1
            print(f"  {name:<25} macro F1 = {f1:.4f}")
        except Exception as e:
            print(f"  {name} failed: {e}")
            results[name] = 0.0

    evaluator.plot_model_comparison(results, save_path=f"{output_dir}/model_comparison.png")

    best = max(results, key=results.get)
    print(f"\n  Best model: {best}  (F1={results[best]:.4f})")
    return best


# ─── Persistence ──────────────────────────────────────────────────────────────

def _save_pipeline(clf, extractor, preprocessor, class_names, output_dir):
    bundle = {
        "classifier":   clf,
        "extractor":    extractor,
        "preprocessor": preprocessor,
        "class_names":  class_names,
    }
    path = f"{output_dir}/pipeline.joblib"
    joblib.dump(bundle, path, compress=3)
    size_mb = os.path.getsize(path) / 1e6
    print(f"\n[SAVE] Pipeline saved → {path}  ({size_mb:.1f} MB)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train document classifier")
    parser.add_argument("--model",   default="svm",
                        choices=list(MODELS.keys()),
                        help="Which model to train (default: svm)")
    parser.add_argument("--feature", default="tfidf",
                        choices=["tfidf", "bert"],
                        help="Feature extraction method (default: tfidf)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all models and plot results")
    parser.add_argument("--output",  default="models",
                        help="Directory for saved model + plots")
    args = parser.parse_args()

    run_pipeline(
        model_key=args.model,
        feature_method=args.feature,
        compare_all=args.compare,
        output_dir=args.output,
    )
