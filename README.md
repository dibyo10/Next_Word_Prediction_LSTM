# 🧠 Next Word Prediction using LSTM (TensorFlow)

A characteristically simple but complete implementation of **Next Word Prediction** using stacked LSTMs in TensorFlow/Keras. This project trains a language model on a structured FAQ corpus and learns to predict the next word given a sequence of previous words.

---

## 📌 Problem Statement

Given an input phrase like:

```
what is
```

The model predicts the most likely next word repeatedly to generate a complete sentence:

```
what is the language spoken by the instructor during the sessions
```

This is classic **language modeling** using sequence learning.

---

## 🏗️ Architecture

The model is implemented using:

- `Tokenizer` — word-level tokenization
- Pre-padding with `pad_sequences`
- One-hot encoded targets
- Stacked LSTM layers

```
Embedding Layer (vocab_size=283, output_dim=100)
          ↓
  LSTM (150 units, return_sequences=True)
          ↓
      LSTM (150 units)
          ↓
  Dense (283 units, softmax)
```

| Component      | Value                   |
|----------------|-------------------------|
| Loss Function  | Categorical Crossentropy |
| Optimizer      | Adam                    |
| Vocabulary Size| 283 words               |
| Max Seq Length | 56 tokens               |

---

## 📊 Training Results

| Metric                  | Value   |
|-------------------------|---------|
| Epochs                  | 100     |
| Final Training Accuracy | ~94%    |
| Final Training Loss     | 0.1692  |

The model transitions from random guessing (~5.53 loss) to confident next-word prediction (~0.16 loss), with a smooth loss curve showing stable convergence.

---

## 📉 Loss Curve

```python
plt.plot(loss)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
```

---

## 🧪 Example Inference

**Input:**
```
what is
```

**Generated Output:**
```
what is the language spoken by the instructor during the sessions
```

The model correctly reconstructs a sentence from the training corpus.

---

## ⚙️ How It Works

### 1️⃣ Tokenization

Each sentence is converted into sequences of incremental n-grams.

**Example — the sentence** `what is the course fee` **becomes:**

```
what is
what is the
what is the course
what is the course fee
```

### 2️⃣ Padding

All sequences are pre-padded to a fixed length to ensure uniform tensor shape.

### 3️⃣ Target Splitting

For each sequence:
- **X** → all tokens except the last
- **y** → last token (one-hot encoded)

### 4️⃣ Training Objective

The model learns the conditional probability:

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

This is standard **autoregressive language modeling**.

---

## 📦 Dependencies

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

**Install all dependencies:**

```bash
pip install tensorflow numpy matplotlib
```

---

## 🚀 How to Run

**Via script:**
```bash
python next_word_prediction_lstm.py
```

**Via notebook:**

Open the `.ipynb` file in Jupyter or Google Colab and run all cells.

---

## 📈 Observations

- Model **overfits** the small dataset — expected, as the corpus is tiny.
- Performs well on memorized structures from the training data.
- This is a **didactic implementation**, not a production-grade language model.

---

## 🔍 Limitations & Improvements

| Limitation                        | Suggested Improvement              |
|-----------------------------------|------------------------------------|
| Small dataset → heavy memorization | Use a larger, diverse corpus       |
| No validation split               | Add train/validation split         |
| No regularization                 | Add Dropout between LSTM layers    |
| Greedy decoding only (argmax)     | Implement Beam Search decoding     |
| Word-level tokenization           | Use subword tokenization (BPE/SentencePiece) |

---

## 🧠 What This Demonstrates

- Sequence modeling fundamentals
- LSTM stacking for deeper representations
- Autoregressive text generation
- End-to-end NLP pipeline (tokenize → train → generate)
- Loss convergence behavior in sequence models
