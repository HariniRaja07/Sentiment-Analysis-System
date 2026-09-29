import streamlit as st
import pickle
import re
import nltk

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Initialize NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Text preprocessing
def preprocess(text):
    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# Load trained model
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)


# Streamlit UI
st.title("AI Sentiment Analysis")

st.write("Enter a sentence or review and analyze its sentiment.")

text = st.text_area(
    "Enter your text:",
    placeholder="Example: I really love this product!"
)


if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        cleaned_text = preprocess(text)

        prediction = model.predict([cleaned_text])[0]

        probabilities = model.predict_proba([cleaned_text])[0]

        confidence = max(probabilities) * 100

        st.subheader("Result")

        if prediction == "positive":
            st.success("Sentiment: POSITIVE")

        elif prediction == "negative":
            st.error("Sentiment: NEGATIVE")

        else:
            st.info("Sentiment: NEUTRAL")

        st.write(f"Confidence: {confidence:.2f}%")
