from flask import Flask, render_template, request
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
import emoji
import unicodedata2
import numpy as np

model_path = "best_model"
loaded_model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


def preprocess_text(text):
    if pd.isna(text):
        return ""

    # Lowercase
    text = text.lower()
    # Normalize unicode characters
    text = unicodedata2.normalize("NFKC", text)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Replace obfuscated characters
    text = (
        text.replace("0", "o")
        .replace("1", "l")
        .replace("!", "i")
        .replace("3", "e")
        .replace("$", "s")
        .replace("5", "s")
        .replace("8", "b")
        .replace("9", "g")
    )
    # Remove username tags but keep certain obfuscations
    text = re.sub(r"@\b(?!ss\b|ggot\b|\$\$\b|f@990t\b)\w+", "", text).replace("@", "a")
    # Remove hashtags
    text = re.sub(r"#\w+", "", text)
    # Decode emojis to text description
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Replace : and _
    text = text.replace(":", " ").replace("_", " ")
    # Handle common issues
    text = text.replace("\n", " ")  # Replace newlines with space
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    text = text.strip()
    return text


def predict(text):
    preprocessed_text = preprocess_text(text)
    encoding = tokenizer(
        [preprocessed_text],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf",
    )
    logits = loaded_model(encoding).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()[0]
    predicted_label = tf.argmax(probabilities).numpy()
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    return label_map[predicted_label], probabilities


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_text = request.form.get("text")
        if input_text:
            predicted_label, probabilities = predict(input_text)
            result = {
                "input_text": input_text,
                "predicted_label": predicted_label,
                "probabilities": probabilities,
            }
            return render_template("index.html", result=result)
    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True, port=8080)
