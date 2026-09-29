import streamlit as st
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


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


with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)


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