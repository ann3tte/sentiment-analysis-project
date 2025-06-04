# Final Project – Comparison of Model Architectures for Sentiment Analysis 🔎
This repository contains our implementation for the Final Project, focused on the comparison of multiple model architectures for sentiment analysis.

## Group Members 
- Alec Ibarra (adi220000)
- Annette Llanas (ajl200006)
- Ashlee Kang (ajk200003)
- Syed Kabir (snk210004)


## Overview
In our Final Project, we implemented and compared four distinct model architectures for sentiment analysis, each representing a different class of machine learning or deep learning approaches:

- Logistic Regression with TF-IDF: A traditional linear model using term frequency–inverse document frequency to convert text into numerical features. It serves as a strong baseline for sentiment classification tasks.

- LSTM with Pretrained Embeddings: A recurrent neural network that leverages GloVe embeddings to model sequential dependencies in text. This architecture captures word order and context, making it well-suited for sentiment detection.
  
- Text CNN: This model uses 1D convolutional filters to capture local patterns in text, such as common word combinations or n-grams. It's a fast and effective approach for sentence-level classification tasks like sentiment analysis.
  
- TinyBERT (Transformer-based Model): A compact transformer model from huawei-noah/TinyBERT_General_4L_312D, designed to be significantly smaller and faster than BERT while maintaining strong performance on NLP tasks. Based on the TinyBERT distillation method, it offers efficient fine-tuning for sentiment analysis without the heavy computational cost of full-sized transformer models.

These models were trained and evaluated on real-world sentiment datasets, and their performances were compared using standard classification metrics. Our goal was to understand the trade-offs between model complexity, training time, and accuracy in sentiment prediction.


## Folders
This repository contains three folders:
- data: Here you can find the data used for this project
- docs: Here you can find the report and slides used for the project
- src: Here you can find the src of the four models tested
