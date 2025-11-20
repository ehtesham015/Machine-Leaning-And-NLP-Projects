import streamlit as st
import joblib
import pandas as pd

# Load model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer (3).pkl")

st.title("📧 Spam/Ham Detection App")

st.write("Enter a message below to classify it as *Spam* or *Ham*.")

# Single text input
user_input = st.text_area("Enter Message Here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        transformed = tfidf.transform([user_input])
        prediction = model.predict(transformed)[0]
        probability = model.predict_proba(transformed)[0]

        label = "Spam 🚫" if prediction == "spam" else "Ham ✅"

        st.subheader(f"Prediction: {label}")
        st.write(f"**Spam Probability:** {probability[1]*100:.2f}%")
        st.write(f"**Ham Probability:** {probability[0]*100:.2f}%")

# Multiple file upload prediction
st.header("📁 Upload CSV File to Test Multiple Messages")

uploaded_file = st.file_uploader("Upload CSV file containing a 'text' column", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("CSV must contain a 'text' column.")
    else:
        # Transform
        transformed_text = tfidf.transform(df["text"])
        df["prediction"] = model.predict(transformed_text)
        df["spam_probability"] = model.predict_proba(transformed_text)[:, 1]

        st.write("### Results:")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
