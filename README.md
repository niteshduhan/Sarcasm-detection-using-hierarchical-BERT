# Sarcasm Detection using Hierarchical BERT

This project implements a Hierarchical BERT-based architecture for sarcasm detection, inspired by the ACL 2020 paper “A Novel Hierarchical BERT Architecture for Sarcasm Detection”. The model leverages contextual information from conversation history to classify a response as sarcastic or non-sarcastic.

#📌 Project Overview

Sarcasm often depends on context rather than isolated sentences. This project addresses that challenge by modeling text in a hierarchical manner:

Word-level encoding using BERT

Sentence-level summarization using CNN

Context-level temporal modeling using BiLSTM

Context–response interaction using convolutional layers

#🧠 Model Architecture

BERT for contextual word embeddings

CNN for sentence and context summarization

BiLSTM to capture temporal dependencies in dialogue

Fully Connected + Sigmoid for final classification

The architecture enables effective utilization of conversation history for sarcasm detection.


#📊 Dataset

The original research evaluates the model on Twitter and Reddit datasets from the FigLang 2020 Shared Task.
This repository focuses on model implementation and experimentation. Dataset files are not included due to licensing constraints.


#🛠 Requirements

Python 3.8+

TensorFlow / PyTorch

HuggingFace Transformers

NumPy, Pandas, Scikit-learn

Jupyter Notebook

#📈 Output

Binary classification: Sarcastic / Non-Sarcastic

Confidence scores using sigmoid probability

Model evaluation using accuracy and F1-score

#⚠️ Disclaimer

This is an independent implementation created for educational and research purposes.
It is inspired by, but not an official reproduction of, the original research paper.

#📚 Reference

Himani Srivastava et al., A Novel Hierarchical BERT Architecture for Sarcasm Detection, ACL 2020.

#👤 Author

Nitesh Duhan
Data Science & NLP Enthusiast
