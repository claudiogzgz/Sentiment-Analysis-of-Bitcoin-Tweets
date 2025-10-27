# Sentiment Analysis of Bitcoin Tweets
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
1. Dictionary-Based Sentiment Analysis (VADER)

Lexicon-based sentiment scoring using NLTK’s VADER to establish a strong baseline for sentiment polarity (positive, negative, neutral).

2. TF-IDF + Logistic Regression Classifier

Tweets are vectorized using TF-IDF representations and classified with a Logistic Regression model for sentiment prediction.

3. RNN Classifier with Custom (Learned) Embeddings

An RNN model trained from scratch using learned embeddings. Includes validation split, early stopping, and regularization to prevent overfitting.

4. RNN Classifier with Pre-trained GloVe Embeddings

An RNN model initialized with GloVe pre-trained embeddings, allowing the network to leverage prior semantic knowledge and improve generalization.

5. Hugging Face Transformer Pipeline (Pre-trained Model)

Applies a pre-trained transformer sentiment model (e.g., DistilBERT) using the Hugging Face pipeline for zero-shot inference on tweets.

6. Fine-tuned DistilBERT Transformer

A DistilBERT model fine-tuned on the tweet dataset using the Hugging Face Trainer API. Inherits base transformer layers and adapts them via one-epoch fine-tuning to evaluate task-specific performance improvements.


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
