# Email Spam Classifier using Support Vector Machine (SVM)

This is a Machine Learning project that classifies emails as Spam or Not Spam using Support Vector Machine and TF-IDF Vectorization. 

## 📌 Project Overview
Spam detection is a binary text classification problem. This project trains an SVM model on an email dataset and provides a web interface to test new email messages.

## ⚙️ Features
- Text preprocessing: lowercase, punctuation removal, stopword removal
- TF-IDF feature extraction
- Support Vector Machine classifier 
- Model and vectorizer saved using joblib

## 🛠️ Tech Stack
- Language: Python 
- Libraries: scikit-learn, pandas, numpy, joblib
- Vectorizer: TfidfVectorizer
- Model: SVC / LinearSVC

## 📊 Algorithm Comparison

I compared 3 algorithms for spam detection on the same dataset:

### 1. Support Vector Machine (SVM) - Model Implemented
- Type: Finds optimal hyperplane to separate spam vs not spam
- Pros: Very effective for high-dimensional text data, good generalization
- Cons: Slower to train on large datasets
- Accuracy on my dataset: 84.62%

### 2. Naive Bayes
- Type: Probabilistic model based on Bayes' Theorem
- Pros: Very fast, works great with text data
- Accuracy on my dataset: 100%

### 3. Logistic Regression
- Type: Linear model for classification
- Pros*: Simple, interpretable, gives probability scores
- Accuracy on my dataset: 77.78%

### Comparison Bar Graph
The bar graph compares Accuracy of all 3 algorithms.

### Conclusion
All 3 algorithms were tested on the email dataset. Naive Bayes achieved 100%, SVM achieved  84.62%, and Logistic Regression achieved 77.78%. SVM performs well on text data but is slower to train compared to NB. Based on accuracy and speed, Naive Bayes was selected as the best model for this task.
