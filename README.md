# Part 3: NLP and Sequence Modeling Mini Project
## Customer Support Sentiment Classification

---

## 📋 Project Overview

This project builds a complete NLP pipeline to classify customer support messages into three sentiment categories: **positive**, **neutral**, and **negative**.

It covers everything from raw text preprocessing through classical ML baselines to a deep learning LSTM model, along with a conceptual discussion of attention mechanisms and transformers.

---

## 📁 Repository Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md                                    ← This file
├── notebook.ipynb                               ← Main Jupyter Notebook (full pipeline)
├── requirements.txt                             ← Python dependencies
├── customer_support_text_classification.csv     ← Dataset (1500 records)
│
└── results/
    ├── dataset_overview.png                     ← Class/channel/length distributions
    ├── baseline_confusion_matrices.png          ← LR and NB confusion matrices
    ├── lstm_training_history.png                ← LSTM accuracy and loss curves
    ├── lstm_confusion_matrix.png                ← LSTM confusion matrix
    ├── model_comparison.png                     ← Side-by-side accuracy/F1 bar chart
    ├── model_evaluation.csv                     ← Per-class metrics for all models
    └── sample_predictions.txt                   ← 15 sample predictions from LSTM
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| File | `customer_support_text_classification.csv` |
| Records | 1,500 |
| Target | `sentiment_label` (positive / neutral / negative) |
| Input Feature | `customer_message` (free text) |
| Channels | email, social, phone, chat, app |
| Avg. Word Count | ~12.7 words/message |
| Missing Values | None |

---

## 🔧 Tasks Completed

### Task 1 — Dataset Understanding
- Record count, class distribution, channel breakdown
- Average text length and word count statistics
- Visualisations: bar chart, pie chart, histogram

### Task 2 — Text Preprocessing
- Lowercasing, special character removal
- Tokenization, stopword removal (custom set, no NLTK dependency)
- Before/after comparison table

### Task 3 — Text Vectorization
- **TF-IDF** (max 3000 features, unigrams + bigrams) for Logistic Regression
- **Bag of Words** for Naive Bayes
- **Tokenizer sequences** (padded to length 30) for LSTM
- Explanation of why text must be numerically encoded

### Task 4 — Baseline Models
- Logistic Regression + TF-IDF
- Multinomial Naive Bayes + Bag of Words
- Metrics: Accuracy, Macro F1, per-class Precision/Recall/F1, Confusion Matrix

### Task 5 — Sequence Model (LSTM)
- Embedding(5000, 64) → SpatialDropout1D → LSTM(64) → Dense(32) → Dense(3, softmax)
- EarlyStopping with patience=4
- Training curves (accuracy & loss per epoch)
- Confusion matrix and classification report

### Task 6 — Attention and Transformer Reflection
- Why RNNs fail on long sequences (vanishing gradient)
- How LSTM gates solve memory retention
- What attention adds to seq-to-seq tasks
- Why transformers power modern Generative AI

---

## 📈 Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression (TF-IDF) | 1.0000 | 1.0000 |
| Naive Bayes (Bag of Words) | 1.0000 | 1.0000 |
| LSTM (Sequences) | 1.0000 | 1.0000 |

> **Note:** Perfect scores are expected for this synthetically generated dataset with consistent language templates. Real-world customer support datasets typically yield 70–85% accuracy.

---

## ⚙️ Installation & Usage

```bash
# 1. Clone or download the repository

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter notebook notebook.ipynb
```

### Requirements
```
pandas==2.2.2
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
tensorflow==2.16.1
```

> Python 3.9+ recommended. No NLTK download required — stopwords are hardcoded.

---

## 🧠 Key Learnings

1. **Text vectorization** is the bridge between raw language and mathematical models.
2. **TF-IDF** captures word importance but ignores word order — fine for short, structured text.
3. **LSTMs** process sequences left-to-right, maintaining memory across words via gated cell states.
4. **Attention** lets models focus on relevant parts of a sequence without the bottleneck of a single hidden vector.
5. **Transformers** parallelise attention across all positions, enabling billion-parameter language models.

---

*Part of AI Mini Project — Part 3: NLP and Sequence Modeling*
