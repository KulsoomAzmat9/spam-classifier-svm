import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC # <-- only change
from sklearn.metrics import accuracy_score, classification_report

st.title("📧 Spam Classifier - SVM")
st.write("Classify messages as Spam or Ham using Support Vector Machine")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('email_data.csv', encoding='latin-1')
    return df

df = load_data()

# Preprocessing
X = df['text']
y = df['label'] .map ({ 'not spam': 0, 'spam': 1})

# Vectorize
tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
X_tfidf = tfidf.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear') # linear works best for text
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.metric("Accuracy", f"{acc*100:.2f}%")
st.text(classification_report(y_test, y_pred))

# Predict new message
st.subheader("Try it yourself")
user_input = st.text_area("Enter a message:")
if st.button("Predict"):
    input_tfidf = tfidf.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    result = "🚨 Spam" if prediction == 1 else "✅ Ham"
    st.success(f"Prediction: {result}")
