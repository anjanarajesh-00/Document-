# Document-
# 📄 Document Classification System

A complete, beginner-friendly ML pipeline for classifying text documents — from raw data to deployed API.

---

## 📁 Project Structure

```
doc_classifier/
├── data/                    # Put your dataset here
│   └── sample_data.py       # Script to generate sample data
├── models/                  # Saved trained models
├── utils/
│   ├── preprocessor.py      # Text cleaning & preprocessing
│   ├── feature_extractor.py # TF-IDF / embedding features
│   └── evaluator.py         # Metrics & confusion matrix
├── tests/
│   └── test_pipeline.py     # Unit tests
├── train.py                 # Full training pipeline
├── predict.py               # Run predictions on new text
├── app.py                   # Flask REST API for deployment
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourname/doc-classifier.git
cd doc-classifier
pip install -r requirements.txt

# 2. Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# 3. Train the model
python train.py

# 4. Predict on new text
python predict.py --text "Your document text here"

# 5. Launch the REST API
python app.py
```

---

## 📊 Supported Categories (20 Newsgroups Demo)

| Category | Examples |
|---|---|
| Technology | computers, electronics, software |
| Sports | hockey, baseball, basketball |
| Politics | government, guns, middle east |
| Science | medicine, space, cryptography |
| Religion | Christianity, atheism, misc.religion |

> Replace with your own dataset — the pipeline is fully generic.

---

## 🧠 Model Options

| Model | Speed | Accuracy | Best For |
|---|---|---|---|
| Logistic Regression + TF-IDF | ⚡⚡⚡ | ⭐⭐⭐ | Baseline, interpretable |
| SVM + TF-IDF | ⚡⚡ | ⭐⭐⭐⭐ | High-dimensional text |
| Random Forest + TF-IDF | ⚡⚡ | ⭐⭐⭐ | Feature importance |
| Gradient Boosting + TF-IDF | ⚡ | ⭐⭐⭐⭐ | Best classic ML |
| BERT Fine-tuned | 🐢 | ⭐⭐⭐⭐⭐ | Production accuracy |

---

## 📈 Evaluation Metrics

- **Accuracy** — overall correct predictions
- **Precision** — of predicted positives, how many are correct
- **Recall** — of actual positives, how many were found
- **F1-Score** — harmonic mean of precision & recall
- **Confusion Matrix** — visualize where the model gets confused

---

## 🔌 API Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "NASA launches new satellite into orbit"}'
```

Response:
```json
{
  "category": "sci.space",
  "confidence": 0.94,
  "all_probabilities": {
    "sci.space": 0.94,
    "sci.med": 0.03,
    "talk.politics.misc": 0.02
  }
}
```

---

## ⚠️ Limitations & Bias

- Model quality depends entirely on training data quality
- May underperform on rare categories (class imbalance)
- Language-specific — English only by default
- BERT requires GPU for fast inference
- Short texts (<20 words) may have low confidence

---

## 🔮 Future Enhancements

- [ ] Multi-label classification (one doc → multiple categories)
- [ ] Multilingual support via XLM-RoBERTa
- [ ] Active learning loop
- [ ] Explainability with LIME/SHAP
- [ ] Docker deployment
- [ ] Streamlit demo UI
