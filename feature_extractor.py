"""
utils/feature_extractor.py
--------------------------
Converts cleaned text into numerical features that ML models can learn from.

THREE METHODS EXPLAINED
-----------------------
1. TF-IDF (Term Frequency–Inverse Document Frequency)
   ➜ Best for: classic ML (SVM, Logistic Regression). Fast, interpretable.
   ➜ Idea: a word is important if it appears often in THIS document but rarely
     across ALL documents (so "the" gets low score, "mitochondria" gets high).

2. Word2Vec / FastText Average Embeddings
   ➜ Best for: capturing semantic meaning ("car" ≈ "automobile").
   ➜ Idea: represent each word as a 300-dim vector, average them per document.
   ➜ Requires a pre-trained model (gensim).

3. BERT Sentence Embeddings
   ➜ Best for: state-of-the-art accuracy when you have GPU / time to spare.
   ➜ Idea: deep transformer that understands context ("bank" → river vs money).
   ➜ Requires transformers + torch. Commented out to keep install lightweight.
"""

import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ─── TF-IDF (recommended default) ────────────────────────────────────────────

class TFIDFExtractor:
    """
    Wraps sklearn's TfidfVectorizer with sensible defaults.

    Parameters
    ----------
    max_features : int    vocabulary size cap (keep top-N by frequency)
    ngram_range  : tuple  (1,1)=unigrams, (1,2)=unigrams+bigrams
    sublinear_tf : bool   apply log(1+tf) instead of raw tf — usually better
    min_df       : int    ignore terms appearing in fewer than N documents
    max_df       : float  ignore terms appearing in more than X% of documents
    """

    def __init__(
        self,
        max_features=50_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
        )
        self._fitted = False

    def fit(self, texts):
        """Learn vocabulary from training texts."""
        logger.info("[TFIDF] Fitting on %d documents...", len(texts))
        self.vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info("[TFIDF] Vocabulary size: %d", vocab_size)
        print(f"[TFIDF] Vocabulary size: {vocab_size:,} terms")
        return self

    def transform(self, texts):
        """Convert texts to TF-IDF sparse matrix."""
        assert self._fitted, "Call fit() first"
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def top_features_per_class(self, clf, class_names, n=15):
        """
        Print the most informative words for each class.
        Works with LinearSVC and LogisticRegression.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        print("\n[TFIDF] Most informative features per class:")
        print("=" * 60)
        for i, class_name in enumerate(class_names):
            if hasattr(clf, "coef_"):
                coef = clf.coef_[i] if clf.coef_.ndim > 1 else clf.coef_[0]
                top_idx = np.argsort(coef)[-n:][::-1]
                top_words = [feature_names[j] for j in top_idx]
                print(f"\n  {class_name}:")
                print(f"    {', '.join(top_words)}")
        print("=" * 60)


# ─── Average Word Embeddings (Word2Vec / GloVe / FastText) ───────────────────

class AverageEmbeddingExtractor:
    """
    Averages pre-trained word vectors to produce a fixed-size doc vector.

    Requires gensim:
        pip install gensim

    Usage:
        extractor = AverageEmbeddingExtractor()
        extractor.load_word2vec("path/to/GoogleNews-vectors-negative300.bin")
        X_train = extractor.fit_transform(train_texts)
    """

    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self._model = None

    def load_word2vec(self, path: str, binary=True):
        """Load a pre-trained Word2Vec binary file."""
        try:
            from gensim.models import KeyedVectors
            print(f"[W2V] Loading embeddings from {path} ...")
            self._model = KeyedVectors.load_word2vec_format(path, binary=binary)
            print(f"[W2V] Loaded {len(self._model):,} word vectors")
        except ImportError:
            raise ImportError("Run: pip install gensim")

    def load_fasttext(self, path: str):
        """Load a FastText .bin model."""
        try:
            import fasttext
            self._model = fasttext.load_model(path)
            self._is_fasttext = True
        except ImportError:
            raise ImportError("Run: pip install fasttext")

    def fit(self, texts):
        """No-op: embeddings are pre-trained, no fitting needed."""
        return self

    def transform(self, texts):
        assert self._model is not None, "Call load_word2vec() first"
        return np.vstack([self._doc_vector(text) for text in texts])

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def _doc_vector(self, text: str) -> np.ndarray:
        tokens = text.split()
        vectors = []
        for token in tokens:
            try:
                vectors.append(self._model[token])
            except KeyError:
                pass  # skip OOV words
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.embedding_dim)


# ─── BERT Sentence Embeddings ─────────────────────────────────────────────────

class BERTExtractor:
    """
    Uses a pre-trained BERT / sentence-transformer model for embeddings.

    Requires:
        pip install sentence-transformers torch

    Usage:
        extractor = BERTExtractor()           # downloads model ~90MB
        X_train = extractor.fit_transform(train_texts)

    NOTES:
    - GPU is ~10x faster: if torch.cuda.is_available() is True it auto-uses GPU
    - 'all-MiniLM-L6-v2' is 6x faster than full BERT with ~5% accuracy loss
    - For production accuracy use 'all-mpnet-base-v2'
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=64, show_progress=True):
        self.model_name    = model_name
        self.batch_size    = batch_size
        self.show_progress = show_progress
        self._model        = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[BERT] Loading '{self.model_name}' ...")
                self._model = SentenceTransformer(self.model_name)
                print(f"[BERT] Model loaded. Embedding dim: {self._model.get_sentence_embedding_dimension()}")
            except ImportError:
                raise ImportError("Run: pip install sentence-transformers torch")

    def fit(self, texts):
        self._load()
        return self

    def transform(self, texts):
        self._load()
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
        )

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ─── Factory function ─────────────────────────────────────────────────────────

def get_extractor(method="tfidf", **kwargs):
    """
    Convenience factory.

    Parameters
    ----------
    method : "tfidf" | "word2vec" | "bert"

    Example
    -------
    extractor = get_extractor("tfidf", max_features=30000)
    """
    method = method.lower()
    if method == "tfidf":
        return TFIDFExtractor(**kwargs)
    elif method in ("word2vec", "w2v", "fasttext"):
        return AverageEmbeddingExtractor(**kwargs)
    elif method == "bert":
        return BERTExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: tfidf | word2vec | bert")
