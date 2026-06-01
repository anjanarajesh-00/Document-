"""
utils/preprocessor.py
---------------------
Step-by-step text cleaning pipeline.

WHAT THIS FILE DOES (plain English):
1. Lowercase everything            → "NASA" == "nasa"
2. Remove URLs, emails, numbers    → reduce noise
3. Remove punctuation              → keep only words
4. Tokenize (split into words)     → ["hello", "world"]
5. Remove stop words               → drop "the", "is", "a" etc.
6. Lemmatize OR stem               → "running" → "run"

WHY EACH STEP MATTERS:
- Without lowercasing: "NASA" and "nasa" are treated as different words.
- Without stop-word removal: common words dominate features and add noise.
- Without lemmatization: "runs", "running", "ran" are 3 separate features.
"""

import re
import string
import logging
from typing import List

import nltk

# Download NLTK resources on first use
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Configurable text preprocessing pipeline.

    Parameters
    ----------
    lowercase      : bool   convert to lowercase
    remove_urls    : bool   strip http/www links
    remove_emails  : bool   strip email addresses
    remove_numbers : bool   strip standalone numbers
    remove_punct   : bool   strip punctuation
    remove_stopwords: bool  drop common English stop words
    extra_stopwords: list   additional words to treat as stop words
    method         : str    "lemmatize" (default) or "stem"
    min_word_len   : int    drop tokens shorter than this (default 2)
    """

    def __init__(
        self,
        lowercase=True,
        remove_urls=True,
        remove_emails=True,
        remove_numbers=True,
        remove_punct=True,
        remove_stopwords=True,
        extra_stopwords=None,
        method="lemmatize",   # "lemmatize" | "stem" | None
        min_word_len=2,
    ):
        self.lowercase        = lowercase
        self.remove_urls      = remove_urls
        self.remove_emails    = remove_emails
        self.remove_numbers   = remove_numbers
        self.remove_punct     = remove_punct
        self.remove_stopwords = remove_stopwords
        self.min_word_len     = min_word_len
        self.method           = method

        # Build stop-word set
        self._stop_words = set(stopwords.words("english"))
        if extra_stopwords:
            self._stop_words.update(extra_stopwords)

        # Normalisation tools
        self._lemmatizer = WordNetLemmatizer()
        self._stemmer    = PorterStemmer()

        # Regex patterns (compiled once for speed)
        self._url_re    = re.compile(r"https?://\S+|www\.\S+")
        self._email_re  = re.compile(r"\S+@\S+\.\S+")
        self._number_re = re.compile(r"\b\d+\b")
        self._punct_re  = re.compile(f"[{re.escape(string.punctuation)}]")

    # ─── Public interface ─────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Return a cleaned string (tokens joined with spaces)."""
        return " ".join(self._process(text))

    def tokenize(self, text: str) -> List[str]:
        """Return a list of processed tokens."""
        return self._process(text)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Vectorised version — process a list of documents."""
        return [self.clean(t) for t in texts]

    # ─── Internal pipeline ────────────────────────────────────────────────────

    def _process(self, text: str) -> List[str]:
        if not isinstance(text, str):
            text = str(text)

        # 1. Lowercase
        if self.lowercase:
            text = text.lower()

        # 2. Remove URLs
        if self.remove_urls:
            text = self._url_re.sub(" ", text)

        # 3. Remove email addresses
        if self.remove_emails:
            text = self._email_re.sub(" ", text)

        # 4. Remove standalone numbers
        if self.remove_numbers:
            text = self._number_re.sub(" ", text)

        # 5. Remove punctuation
        if self.remove_punct:
            text = self._punct_re.sub(" ", text)

        # 6. Tokenize
        tokens = word_tokenize(text)

        # 7. Remove stop words
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        # 8. Drop very short tokens
        tokens = [t for t in tokens if len(t) >= self.min_word_len]

        # 9. Lemmatize or stem
        if self.method == "lemmatize":
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]
        elif self.method == "stem":
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    # ─── Debug helper ─────────────────────────────────────────────────────────

    def explain(self, text: str):
        """Print a step-by-step breakdown of what happens to one document."""
        print("=" * 60)
        print("PREPROCESSING WALKTHROUGH")
        print("=" * 60)

        steps = {}
        t = text

        steps["0_original"] = t[:200]

        if self.lowercase:
            t = t.lower()
            steps["1_lowercase"] = t[:200]

        if self.remove_urls:
            t = self._url_re.sub(" ", t)
            steps["2_no_urls"] = t[:200]

        if self.remove_emails:
            t = self._email_re.sub(" ", t)
            steps["3_no_emails"] = t[:200]

        if self.remove_numbers:
            t = self._number_re.sub(" ", t)
            steps["4_no_numbers"] = t[:200]

        if self.remove_punct:
            t = self._punct_re.sub(" ", t)
            steps["5_no_punct"] = t[:200]

        tokens = word_tokenize(t)
        steps["6_tokenized"] = tokens[:20]

        if self.remove_stopwords:
            tokens = [tok for tok in tokens if tok not in self._stop_words]
            steps["7_no_stopwords"] = tokens[:20]

        tokens = [tok for tok in tokens if len(tok) >= self.min_word_len]
        steps["8_min_length"] = tokens[:20]

        if self.method == "lemmatize":
            tokens = [self._lemmatizer.lemmatize(tok) for tok in tokens]
            steps["9_lemmatized"] = tokens[:20]
        elif self.method == "stem":
            tokens = [self._stemmer.stem(tok) for tok in tokens]
            steps["9_stemmed"] = tokens[:20]

        for step_name, value in steps.items():
            print(f"\n[{step_name}]")
            print(f"  {value}")

        print("\n" + "=" * 60)
