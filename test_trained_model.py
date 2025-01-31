import os
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
import numpy as np
import emoji
import unicodedata2
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


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
fig_dir = "figures"
max_length = 128
batch_size = 32
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


def prepare_data(df, text_column):
    """Prepare and preprocess the dataset."""
    # Log some examples of preprocessed text

    # Apply text preprocessing
    df[text_column] = df[text_column].apply(preprocess_text)

    # Remove empty strings and reset index
    df = df[df[text_column].str.len() > 0].reset_index(drop=True)

    # Check data distribution
    analyse_data_distribution(df[text_column], df["label"])

    return df


def analyse_data_distribution(texts, labels):
    """Check label distribution"""
    total = len(labels)
    hate_speech = sum(labels == 0)
    offensive_language = sum(labels == 1)
    neutral = sum(labels == 2)

    print(f"Total examples: {total}")
    print(f"Hate speech: {hate_speech} ({(hate_speech/total)*100:.2f}%)")
    print(
        f"Offensive language: {offensive_language} ({(offensive_language/total)*100:.2f}%)"
    )
    print(f"Not hate: {neutral} ({(neutral/total)*100:.2f}%)")

    # Check text lengths
    lengths = [len(str(t)) for t in texts]
    print(f"\nAverage text length: {sum(lengths)/len(lengths):.2f}")
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")

    # Sample a few examples from each class
    print("\nSample hate speech:")
    for t, l in zip(texts, labels):
        if l == 0:
            print(f"- {t}")
            break

    # Sample a few examples from each class
    print("\nSample offensive language:")
    for t, l in zip(texts, labels):
        if l == 1:
            print(f"- {t}")
            break

    print("\nSample non-hate speech:")
    for t, l in zip(texts, labels):
        if l == 2:
            print(f"- {t}")
            break


def evaluate(test_df, text_column, label_column):
    """Evaluate the model on test data."""

    # Prepare test data
    test_df = prepare_data(test_df, text_column)
    test_ds = tokenize_data(test_df[text_column], test_df[label_column])
    test_ds = test_ds.batch(batch_size)

    # Make predictions
    predictions = []
    true_labels = []
    all_probs = []

    for batch in test_ds:
        logits = loaded_model(batch[0])[0]
        probs = tf.nn.softmax(logits, axis=1).numpy()
        pred = tf.argmax(logits, axis=1).numpy()

        predictions.extend(pred)
        true_labels.extend(tf.argmax(batch[1], axis=1).numpy())
        all_probs.extend(probs)

    # Probabilities for ROC AUC, convert to numpy array
    y_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision": precision_score(true_labels, predictions, average="weighted"),
        "recall": recall_score(true_labels, predictions, average="macro"),
        "f1": f1_score(true_labels, predictions, average="weighted"),
        "roc_auc": roc_auc_score(true_labels, y_probs, multi_class="ovr"),
    }

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    plot_confusion_matrix(cm)

    return metrics


def tokenize_data(texts, labels):
    """Tokenize text data for model input."""
    tokenizer = BertTokenizer.from_pretrained(model_path)

    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="tf",
    )

    # Convert labels to one-hot encoded format
    labels = tf.keras.utils.to_categorical(labels, num_classes=3)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "token_type_ids": encodings["token_type_ids"],
            },
            labels,
        )
    )
    return dataset


def plot_confusion_matrix(cm):

    print("Plotting confusion matrix...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix\n{text_data}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    filename = os.path.join(
        fig_dir, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    )
    plt.savefig(filename)
    plt.close()


metrics = evaluate(test_df=text_data, text_column="text", label_column="label")
print("Evaluation Metrics:", metrics)

print("Extra testing:")
test_text = [
    "I am going to 🔪 some 👳🏾‍♂️",
    "I am going to love some 👳🏾‍♂️",
    "👳🏾‍♂️ is 💩",
    "🏳️‍🌈 ❤️",
    "🧑🏿‍🦲🤢",
    "𝕟𝕚𝕘𝕘𝕖𝕣",
    "ᶠᵁᶜᴷ ʸᴼᵁ 𝒸𝒽𝒾𝓃𝓀 ",
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
