# Sarcasm Detection — Hierarchical BERT

> **72.05% test accuracy** on Reddit dialogue data using a context-aware BERT + CNN + BiLSTM stack, outperforming flat-BERT baselines on conversational sarcasm.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red?style=flat-square&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-latest-green?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)

---

## Overview

Sarcasm in dialogue is context-dependent — classifying it at the sentence level alone fails. This project implements a **hierarchical deep learning architecture** that models conversation history across three encoding levels: word → sentence → context.

Inspired by *Srivastava et al., ACL 2020 — "A Novel Hierarchical BERT Architecture for Sarcasm Detection"*.

| Metric | Value |
|--------|-------|
| Test Accuracy | **72.05%** |
| Task | Binary Classification (Sarcastic / Non-Sarcastic) |
| Output | Sigmoid probability score |
| Dataset | Reddit dialogue threads (Kaggle) |

---

## Architecture

```
Input: [context utterances] + [target response]
          │
     ┌────▼────┐
     │  BERT   │  ← Contextual word-level embeddings
     └────┬────┘
          │
     ┌────▼────┐
     │   CNN   │  ← Sentence-level summarization
     └────┬────┘
          │
     ┌────▼────┐
     │ BiLSTM  │  ← Temporal context modeling across utterances
     └────┬────┘
          │
     ┌────▼─────────┐
     │  Conv Layer  │  ← Context–response interaction
     └────┬─────────┘
          │
     ┌────▼──────────────┐
     │  FC + Sigmoid     │  ← Binary classification
     └───────────────────┘
```

**Why this works over flat BERT:**  
Flat BERT truncates or ignores dialogue history. This stack encodes each utterance independently via BERT, then models the *temporal flow* of conversation via BiLSTM before comparing against the final response — preserving sarcasm cues that live in context, not isolated text.

---

## Methodology

1. **Data** — Reddit sarcasm corpus (Kaggle); multi-turn dialogue format with labeled response
2. **Preprocessing** — Tokenization via BERT tokenizer, conversation chunking, label encoding
3. **Word Encoding** — `bert-base-uncased` embeddings per utterance
4. **Sentence Encoding** — CNN with max-pooling across token embeddings → fixed-size utterance vector
5. **Context Encoding** — BiLSTM over ordered utterance vectors → dialogue-aware context representation
6. **Interaction** — Convolutional layer merging context and response representations
7. **Classification** — Fully connected layer + Sigmoid; binary cross-entropy loss
8. **Evaluation** — Accuracy on held-out test split; sigmoid confidence scores

---

## Results

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **Hierarchical BERT (ours)** | **72.05%** | BERT + CNN + BiLSTM |
| Flat BERT baseline | ~67–69% | Single-sentence classification |
| LSTM (no BERT) | ~63–65% | No pretrained embeddings |

> *Baseline figures are reference estimates from the ACL 2020 paper context.*

---

## Project Structure

```
sarcasm-detection-hierarchical-bert/
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modelling_Evaluation.ipynb
│
├── src/
│   ├── model.py          # Hierarchical BERT architecture
│   ├── dataset.py        # DataLoader + tokenization
│   ├── train.py          # Training loop
│   └── evaluate.py       # Inference + metrics
│
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/niteshduhan/sarcasm-detection-hierarchical-bert.git
cd sarcasm-detection-hierarchical-bert

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data
# Place train.csv / val.csv / test.csv in data/

# 4. Train the model
python src/train.py --epochs 10 --batch_size 16 --lr 2e-5

# 5. Evaluate
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

**requirements.txt**
```
transformers>=4.30.0
torch>=1.13.0
tensorflow>=2.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
```

---

## Key Takeaways

- **Hierarchical encoding beats flat classification** — modeling each utterance separately before aggregating via BiLSTM preserves inter-sentence sarcasm signals that a single BERT pass collapses.
- **CNN as utterance summarizer** — using CNN over BERT token embeddings produces compact, semantically dense sentence vectors with lower compute cost than pooling strategies.
- **Context window matters** — sarcasm detection accuracy is directly correlated with the number of preceding utterances included; truncating to 1–2 turns degrades performance measurably.

---

## Reference

Himani Srivastava et al., *A Novel Hierarchical BERT Architecture for Sarcasm Detection*, ACL 2020.  
[Paper Link](https://aclanthology.org/2020.figlang-1.14/)

> **Disclaimer:** Independent research implementation. Not an official reproduction of the original paper.

---

## Author

**Nitesh Duhan**  
M.S. Data Science | NLP & Deep Learning  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-niteshduhan--carp112-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/niteshduhan-carp112)
[![Email](https://img.shields.io/badge/Email-niteshduhan686@gmail.com-red?style=flat-square&logo=gmail)](mailto:niteshduhan686@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-niteshduhan-black?style=flat-square&logo=github)](https://github.com/niteshduhan)
