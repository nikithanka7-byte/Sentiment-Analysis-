import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Sentiment Analysis")
st.write("Enter your text and find whether it is Positive, Negative, or Neutral.")

st.divider()

@st.cache_resource
def get_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

sentiment_model = get_model()

text = st.text_area(
    "Enter your text:",
    placeholder="Example: The weather is okay today."
)

if st.button("Analyze Sentiment"):

    if text.strip() == "":
        st.warning("Please enter some text.")

    else:
        result = sentiment_model(text)[0]

        label = result["label"]
        score = result["score"]

        st.subheader("Result")

        if label.upper() == "POSITIVE":
            st.success("😊 Positive")

        elif label.upper() == "NEGATIVE":
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")

        st.write(f"Confidence: **{score:.2%}**")


st.divider()
st.caption("✨ Created by Nikitha R ✨")