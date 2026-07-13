import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

st.title("📧 Email Spam Classifier")
st.write("Type a message and I'll tell you if it's SPAM or NOT SPAM")

@st.cache_data
def load_data():
    df = pd.read_csv('email_data.csv', encoding='latin-1')
    # Fix labels: handle ALL CAPS, spaces
    df['label'] = df['label'].str.strip().str.lower().map({'not spam': 0, 'spam': 1})
    df = df.dropna() # remove any NaN rows
    return df

df = load_data()

X = df['text']
y = df['label']

# Use TF-IDF instead of CountVectorizer for better accuracy
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear', C=10, probability=True)
model.fit(X_train, y_train)

# Show accuracy
acc = accuracy_score(y_test, model.predict(X_test))
st.sidebar.metric("Model Accuracy", f"{acc*100:.2f}%")

st.divider()

# User Input
user_input = st.text_area("Enter email/message to classify:", height=150)

if st.button("Classify Message", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first")
    else:
        # Transform user input
        user_vec = vectorizer.transform([user_input])
        prediction = model.predict(user_vec)
        prob = model.predict_proba(user_vec)[0]

        if prediction[0] == 1:
            st.error(f"🚨 SPAM")
            st.write(f"Confidence: {prob[1]*100:.2f}%")
        else:
            st.success(f"✅ NOT SPAM")
            st.write(f"Confidence: {prob[0]*100:.2f}%")

st.divider()
st.caption("Trained on your email_data.csv using SVM + TF-IDF")
