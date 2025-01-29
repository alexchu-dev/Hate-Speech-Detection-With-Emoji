from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
import emoji
import unicodedata2


# Preprocessing function
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


# Define custom loss function
def weighted_loss(y_true, y_pred):
    weights = tf.constant([3.4, 1.0, 2.3])  # Adjust weights as needed
    y_true = tf.cast(y_true, tf.int32)
    sample_weights = tf.gather(weights, y_true)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weighted_loss = loss * sample_weights
    return tf.reduce_mean(weighted_loss)


# Load the saved model
model_path = "best_model"
loaded_model = TFBertForSequenceClassification.from_pretrained(model_path)
print("Model loaded successfully!")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)


# Prediction function
def predict(texts, tokenizer, model, max_length=128):
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]

    # Tokenize the inputs
    encodings = tokenizer(
        processed_texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )

    # Get predictions
    logits = model(encodings).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predicted_labels = tf.argmax(probabilities, axis=1).numpy()

    return processed_texts, predicted_labels, probabilities


# Example usage: Predict single or multiple texts
text_data = pd.read_csv("datasets/test_data_added_emoji.csv")
texts_to_predict = text_data["text"].tolist()
text_data_original_label = text_data["label"].tolist()

# Get predictions
processed_texts, predictions, probabilities = predict(
    texts_to_predict, tokenizer, loaded_model
)

# Print predictions
label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
for text, original_label, processed, label, prob in zip(
    texts_to_predict,
    text_data_original_label,
    processed_texts,
    predictions,
    probabilities,
):
    print(f"Original Text: {text}")
    print(f"Processed Text: {processed}")
    print(f"Original Label: {label_map[original_label]}")
    print(f"Predicted Label: {label_map[label]}")
    print(f"Probabilities: {prob}\n")

test_text = [
    "🔪🧑🏿‍🦲",
    "👳🏾‍♂️ is 💩",
    "love ❤️",
    "🧑🏿‍🦲🤢",
    "𝔽𝕌ℂ𝕂 𝕐𝕆𝕌",
    "ᶠᵁᶜᴷ ʸᴼᵁ",
    "I ❤️ you",
]
processed_texts, predictions, probabilities = predict(
    test_text, tokenizer, loaded_model
)
for text, processed, label, prob in zip(
    test_text, processed_texts, predictions, probabilities
):
    print(f"Original Text: {text}")
    print(f"Processed Text: {processed}")
    print(f"Predicted Label: {label_map[label]}")
    print(f"Probabilities: {prob}\n")
