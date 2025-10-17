import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.nn.functional import softmax
import torch

# Load model
@st.cache_resource
def load_model():
    model_path = "sentiment_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="Movie Review Sentiment", page_icon="🎬", layout="centered")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to see if it’s Positive or Negative!")

user_input = st.text_area("Your review:")

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256)
        outputs = model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        sentiment = "🌟 Positive" if pred == 1 else "😠 Negative"
        confidence = probs[0][pred].item()

        if pred == 1:
            st.success(f"{sentiment} (Confidence: {confidence:.2f})")
        else:
            st.error(f"{sentiment} (Confidence: {confidence:.2f})")

