# Sentiment-Analysis-of-Bitcoin-Tweets
Comprehensive sentiment analysis of Bitcoin-related tweets using multiple Natural Language Processing (NLP) techniques. From lexicon-based methods to deep learning and transformer models.


## Project Overview

This project performs sentiment analysis on Bitcoin tweets using a variety of NLP approaches.
The goal is to compare traditional machine learning, recurrent neural networks, and transformer-based models to determine which performs best for binary sentiment classification (positive or negative).

The dataset includes 2,000 labelled tweets, split into 1,500 for training and 500 for testing, each assigned a sentiment score (1 = positive, 0 = negative).


## Environment Setup
	•	Python version: 3.12.2
	•	All dependencies are listed in requirements.txt
	•	External resources (e.g., GloVe embeddings, Hugging Face models) are automatically downloaded when the notebook runs


## Implemented Approaches

1. Text Preprocessing

Tweets are cleaned and normalized by:
	•	Removing HTML tags, links, and Unicode artifacts
	•	Lowercasing and tokenizing
	•	Optionally applying stopword removal and lemmatization

2. Dictionary-Based Sentiment Analysis (VADER)

Lexicon-based sentiment scoring using NLTK’s VADER to establish a baseline.

3. TF-IDF + Logistic Regression
	•	Transform text using TF-IDF vectorization
	•	Train a Logistic Regression classifier for sentiment prediction

4. RNN Classifiers
	•	Custom Embeddings: RNN trained from scratch with learned embeddings
	•	Pre-trained Embeddings: RNN initialized with GloVe vectors
	•	Includes validation split and early stopping

5. Transformer Models
	•	Apply pre-trained transformer models (e.g., DistilBERT) via Hugging Face pipelines
	•	Optionally fine-tune a lightweight transformer for one epoch to evaluate performance improvements


 ## Evaluation

All models are evaluated on the labelled test set using standard binary classification metrics:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-Score
	•	ROC-AUC

A summary comparison table is included at the end of the notebook to benchmark performance across all approaches.


## Technologies & Libraries
	•	Python, NumPy, Pandas
	•	Scikit-learn — TF-IDF, Logistic Regression, evaluation metrics
	•	TensorFlow / Keras — RNN models and training
	•	GloVe — Pre-trained word embeddings
	•	NLTK — VADER sentiment analysis and preprocessing
	•	Hugging Face Transformers — DistilBERT and sentiment pipelines
	•	Matplotlib, Seaborn — Data visualization


## Contact

Created by [Claudio Gonzalez](https://github.com/claudiogzgz)  
Feel free to connect or reach out via GitHub if you have questions or ideas!
