"""
predict.py
----------
Load a saved pipeline and classify new text.

Usage:
    python predict.py --text "NASA discovers new exoplanet near Proxima Centauri"
    python predict.py --file my_docs.txt
    python predict.py --interactive
"""

import argparse
import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(__file__))


def load_pipeline(model_dir="models"):
    path = os.path.join(model_dir, "pipeline.joblib")
    if not os.path.exists(path):
        print(f"[ERROR] No saved model found at '{path}'")
        print("        Run 'python train.py' first to train the model.")
        sys.exit(1)
    bundle = joblib.load(path)
    return bundle


def predict_one(text: str, bundle: dict) -> dict:
    """
    Classify a single document.

    Returns dict with:
        category       — top predicted class
        confidence     — probability of top class
        all_probs      — dict of {class: probability}
    """
    preprocessor = bundle["preprocessor"]
    extractor    = bundle["extractor"]
    clf          = bundle["classifier"]
    class_names  = bundle["class_names"]

    # 1. Preprocess
    cleaned = preprocessor.clean(text)

    # 2. Extract features
    features = extractor.transform([cleaned])

    # 3. Predict
    predicted_label = clf.predict(features)[0]

    # 4. Probabilities (if available)
    all_probs = {}
    try:
        proba = clf.predict_proba(features)[0]
        all_probs = dict(sorted(zip(class_names, proba), key=lambda x: -x[1]))
        confidence = max(proba)
    except AttributeError:
        confidence = 1.0
        all_probs  = {predicted_label: 1.0}

    return {
        "category":      predicted_label,
        "confidence":    round(float(confidence), 4),
        "all_probs":     {k: round(float(v), 4) for k, v in all_probs.items()},
    }


def predict_many(texts: list, bundle: dict) -> list:
    """Batch prediction for a list of texts."""
    return [predict_one(t, bundle) for t in texts]


def interactive_mode(bundle):
    print("\n📄  Document Classifier — Interactive Mode")
    print("   Type your text and press Enter. Type 'quit' to exit.\n")
    while True:
        text = input("Text> ").strip()
        if not text or text.lower() in ("quit", "exit", "q"):
            break
        result = predict_one(text, bundle)
        _pretty_print(text, result)


def _pretty_print(text, result):
    print("\n" + "─" * 60)
    print(f"  Input     : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"  Category  : {result['category']}")
    print(f"  Confidence: {result['confidence'] * 100:.1f}%")
    print("\n  All probabilities:")
    for cat, prob in result["all_probs"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cat:<35} {prob*100:>5.1f}%  {bar}")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict document category")
    parser.add_argument("--text",        type=str,  help="Single text to classify")
    parser.add_argument("--file",        type=str,  help="Text file with one doc per line")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--model-dir",   default="models")
    args = parser.parse_args()

    bundle = load_pipeline(args.model_dir)
    print(f"[PREDICT] Model loaded. Classes: {bundle['class_names']}")

    if args.text:
        result = predict_one(args.text, bundle)
        _pretty_print(args.text, result)

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[PREDICT] Classifying {len(lines)} documents...")
        for line in lines:
            result = predict_one(line, bundle)
            _pretty_print(line, result)

    elif args.interactive:
        interactive_mode(bundle)

    else:
        # Demo with sample texts
        demos = [
            "NASA announces plans for a crewed mission to Mars in 2035",
            "The team scored three goals in the second half to win the championship",
            "New research shows aspirin may reduce the risk of certain cancers",
            "Congress passes new legislation on firearm background checks",
            "The Archbishop of Canterbury delivered a sermon on forgiveness",
        ]
        print("\n[DEMO] Running predictions on sample texts:\n")
        for text in demos:
            result = predict_one(text, bundle)
            _pretty_print(text, result)
