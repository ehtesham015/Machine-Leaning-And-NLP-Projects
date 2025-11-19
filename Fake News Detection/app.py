import streamlit as st
import joblib

model = joblib.load("fake_news_model (1).pkl")
tfidf = joblib.load("tfidf_vectorizer (1).pkl")

st.title("Fake News Detection App")

input_text = st.text_area("Enter News Article:")

if st.button("Predict"):
    transformed = tfidf.transform([input_text])
    result = model.predict(transformed)[0]

    if result == 0:
        st.error("🚨 Fake News")
    else:
        st.success("✔ Real News")
