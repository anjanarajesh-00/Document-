"""
tests/test_pipeline.py
-----------------------
Basic unit tests to verify the pipeline works correctly.

Run:
    python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from utils.preprocessor      import TextPreprocessor
from utils.feature_extractor import TFIDFExtractor


# ─── Preprocessor tests ───────────────────────────────────────────────────────

class TestPreprocessor:

    def setup_method(self):
        self.pp = TextPreprocessor()

    def test_lowercase(self):
        result = self.pp.clean("NASA Is GREAT")
        assert result == result.lower()

    def test_removes_urls(self):
        result = self.pp.clean("Visit https://example.com for more info")
        assert "http" not in result
        assert "example" not in result

    def test_removes_emails(self):
        result = self.pp.clean("Contact us at support@company.com")
        assert "@" not in result

    def test_removes_stopwords(self):
        result = self.pp.tokenize("this is a great test")
        assert "this" not in result
        assert "is"   not in result
        assert "a"    not in result

    def test_lemmatization(self):
        pp = TextPreprocessor(method="lemmatize")
        tokens = pp.tokenize("running dogs are barking")
        # lemmatize should reduce "running" → "running" (verb) or "run"
        # and "dogs" → "dog"
        assert "dog" in tokens

    def test_stemming(self):
        pp = TextPreprocessor(method="stem")
        tokens = pp.tokenize("running dogs")
        # Porter stemmer: "running" → "run"
        assert any("run" in t for t in tokens)

    def test_empty_string(self):
        result = self.pp.clean("")
        assert isinstance(result, str)

    def test_non_string_input(self):
        result = self.pp.clean(None)
        assert isinstance(result, str)

    def test_batch(self):
        texts = ["Hello world", "foo bar baz"]
        results = self.pp.clean_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)


# ─── Feature extractor tests ──────────────────────────────────────────────────

class TestTFIDFExtractor:

    def setup_method(self):
        self.extractor = TFIDFExtractor(max_features=1000)
        self.corpus = [
            "machine learning is fascinating",
            "deep neural networks learn patterns",
            "basketball team scores winning goal",
            "hockey player scores in overtime",
            "nasa launches new rocket mission",
        ]

    def test_fit_transform_shape(self):
        X = self.extractor.fit_transform(self.corpus)
        assert X.shape[0] == len(self.corpus)
        assert X.shape[1] <= 1000

    def test_transform_new_texts(self):
        self.extractor.fit(self.corpus)
        new_texts = ["space shuttle", "football game"]
        X = self.extractor.transform(new_texts)
        assert X.shape[0] == 2

    def test_fit_required_before_transform(self):
        extractor = TFIDFExtractor()
        with pytest.raises(AssertionError):
            extractor.transform(["some text"])

    def test_consistent_features(self):
        X_train = self.extractor.fit_transform(self.corpus)
        X_test  = self.extractor.transform(["machine learning model"])
        assert X_train.shape[1] == X_test.shape[1]


# ─── Integration test ─────────────────────────────────────────────────────────

class TestIntegration:
    """End-to-end mini pipeline test."""

    def test_full_pipeline(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # Tiny dataset
        texts = [
            "soccer football match goal player",
            "basketball court hoop dribble game",
            "tennis racket serve volley match",
            "hockey puck ice skate goal net",
            "python programming code function",
            "java software developer algorithm",
            "machine learning neural network data",
            "javascript web react html css",
        ] * 5   # repeat so we have enough training data

        labels = (
            ["sports"] * 4 + ["tech"] * 4
        ) * 5

        pp = TextPreprocessor()
        ext = TFIDFExtractor(max_features=500)
        clf = LogisticRegression(max_iter=100, class_weight="balanced")

        from sklearn.model_selection import train_test_split
        X_clean = pp.clean_batch(texts)
        X_train, X_test, y_train, y_test = train_test_split(X_clean, labels, test_size=0.25, random_state=0)

        X_tr = ext.fit_transform(X_train)
        X_te = ext.transform(X_test)
        clf.fit(X_tr, y_train)

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_test, y_pred)

        # With clean, separable data we expect > 80% accuracy
        assert acc >= 0.75, f"Integration test accuracy too low: {acc:.2f}"
        print(f"\n  Integration test accuracy: {acc:.2f} ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
