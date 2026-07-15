import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC # <-- Only change 1
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📧 Spam Email Classifier - SVM")

# 1. LOAD DATA
df = pd.read_csv("email_data.csv")
st.success("✅ Loaded email_data.csv successfully!")

# 2. SHOW DATA + CLEAN LABELS
st.subheader("1. Dataset Information")
st.dataframe(df.head())

df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

# SAME CLEANING FOR SPAM / NOT SPAM
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['label'] = df['label'].map({'not spam': 0, 'spam': 1})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

st.subheader("2. Training Model...")
st.write(f"*Total emails used:* {len(df)}")

# 3. TRAIN
X = df['text']
y = df['label']
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = SVC() # <-- Only change 2. Using default SVM
model.fit(X_train, y_train)

joblib.dump(model, 'spam_model_svm.joblib')
joblib.dump(vectorizer, 'vectorizer_svm.joblib')
st.success("✅ Model Trained and Saved!")

# 4. RESULTS
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("3. Model Performance")
st.metric("Accuracy", f"{acc*100:.2f}%")
st.text(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

# 5. PREDICTION
