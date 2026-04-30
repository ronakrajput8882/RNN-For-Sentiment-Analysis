<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🎬%20IMDB%20Sentiment%20Analysis&fontSize=42&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Binary%20Text%20Classification%20using%20TF-IDF%20%2B%20PyTorch%20RNN&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-0769AD?style=for-the-badge)](https://nltk.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Project Overview

A complete **NLP sentiment analysis pipeline** trained on the **IMDB Movie Reviews dataset** — classifying 50,000 reviews as Positive or Negative using classical text preprocessing, TF-IDF feature engineering, and a PyTorch-based RNN neural model.

| Property | Details |
|:---|:---|
| Task | Binary Text Classification |
| Domain | Natural Language Processing (NLP) |
| Dataset | IMDB Movie Reviews |
| Model | PyTorch RNN + TF-IDF |
| Output | ✅ Positive / ❌ Negative |

---

## 📂 Dataset

| Property | Details |
|:---|:---|
| Source | IMDB Movie Reviews |
| Total Samples | 50,000 |
| Classes | Positive / Negative |
| Class Balance | Equal (25K each) |
| TF-IDF Features | 5,000 |

---

## 🔄 Pipeline Workflow

```
Raw Text → Cleaning → Tokenization → Stopword Removal → Stemming → TF-IDF → PyTorch RNN → Prediction
```

### 1️⃣ Text Cleaning
- Lowercasing all characters
- Removing URLs, HTML tags, and punctuation
- Stripping special characters and noise

### 2️⃣ Tokenization & Stopword Removal
- Word-level tokenization via NLTK
- Filtering English stopwords (NLTK corpus)

### 3️⃣ Stemming
- `PorterStemmer` — reduces words to root form
- Reduces vocabulary size and improves generalization

### 4️⃣ TF-IDF Vectorization
- Converts cleaned text → numerical vectors
- Max features: **5,000**
- Captures term importance relative to corpus

### 5️⃣ PyTorch RNN Model
- Input: 5000-dim TF-IDF vector
- RNN layer with `hidden_size=128`
- Fully Connected → Sigmoid → Binary output

---

## 🤖 Model Architecture ⭐ Best Model

```
Input Layer (5000-dim TF-IDF vector)
        ↓
  RNN Layer (hidden_size=128)
        ↓
  Fully Connected Layer
        ↓
    Sigmoid Activation
        ↓
  Binary Output (0=Negative, 1=Positive)
```

**Training Configuration:**

| Parameter | Value |
|:---|:---|
| Loss Function | BCELoss |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 64 |
| Decision Threshold | 0.5 |

> ⚠️ **Note:** TF-IDF removes sequential information from text, so the RNN layer effectively behaves as a dense layer. For true sequence modelling, use token-indexed inputs with embeddings.

---

## 📊 Results

| Metric | Value |
|:---|:---:|
| **Accuracy** | **~87–89%** |
| Loss Function | BCELoss |
| Decision Threshold | 0.5 |
| Classes | Positive / Negative |

> 📝 Replace with your actual epoch-wise accuracy after training.

---

## 🔍 Key Insights

- 🧹 **Preprocessing matters** — removing HTML tags, URLs, and stopwords significantly improves TF-IDF signal quality
- 📉 **TF-IDF + RNN** is a hybrid approach: TF-IDF discards word order, making the RNN layer function like a fully connected dense layer
- 🏆 **Simple models are competitive** — TF-IDF with a lightweight neural network can match more complex architectures on balanced datasets
- 💡 **Stemming reduces vocabulary** by ~30–40%, leading to more generalizable feature representations
- ⚠️ No embeddings (Word2Vec / GloVe) means the model misses semantic word relationships — a known limitation

---

## 🗂️ Repository Structure

```
IMDB-Sentiment-Analysis/
│
├── rnn_sentiment_analysis.py   # Main training & inference script
├── preprocessing.py            # Text cleaning & TF-IDF pipeline
├── model.py                    # PyTorch RNN model definition
├── evaluate.py                 # Accuracy evaluation script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/ronakrajput8882/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis

# Install dependencies
pip install pandas numpy nltk scikit-learn torch

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Run the training & evaluation script
python rnn_sentiment_analysis.py
```

---

## 🧠 Key Learnings

- TF-IDF is fast, interpretable, and surprisingly effective for binary text classification
- Vanilla RNN without proper sequential input doesn't leverage its recurrent strength — LSTM/GRU + embeddings would be the true upgrade
- Balanced datasets (equal class distribution) simplify model calibration and threshold tuning
- Text preprocessing quality directly determines TF-IDF feature quality
- BCELoss + Adam is a reliable baseline for binary classification with sigmoid output

---

## ⚠️ Limitations

- Not a true sequence model — TF-IDF removes word order
- No word embeddings (Word2Vec, GloVe, FastText)
- No GPU optimization or mixed precision training
- No hyperparameter tuning (learning rate, hidden size, etc.)

---

## 🚀 Future Improvements

- [ ] Replace TF-IDF with tokenized sequences + embedding layer
- [ ] Use LSTM / GRU instead of vanilla RNN
- [ ] Add pre-trained word embeddings (GloVe / FastText)
- [ ] Fine-tune a Transformer model (DistilBERT / BERT)
- [ ] Add training curves and confusion matrix visualization

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| Python 3.10+ | Core language |
| PyTorch | RNN model training & inference |
| Scikit-Learn | TF-IDF vectorization |
| NLTK | Tokenization, stopwords, stemming |
| Pandas | Data loading & management |
| NumPy | Numerical operations |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>