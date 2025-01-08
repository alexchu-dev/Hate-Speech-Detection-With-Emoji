from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import re
import emoji
import unicodedata2

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata2.normalize('NFKC', text)
    # Normalize emojis
    text = emoji.demojize(text, delimiters=(" ", " "))
    # Replace : and _
    text = text.replace(":", " ").replace("_", " ")
    
    print("After emoji demojize: ", text)
    # Replace obfuscated characters
    text = text.replace('0', 'o').replace('1', 'i').replace('3', 'e').replace('$', 's')
    text = re.sub(r'[!@#$%^&*]', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Define custom loss function
def weighted_loss(y_true, y_pred):
    weights = tf.constant([6.0, 1.0, 3.0])  # Adjust weights as needed
    y_true = tf.cast(y_true, tf.int32)
    sample_weights = tf.gather(weights, y_true)
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    weighted_loss = loss * sample_weights
    return tf.reduce_mean(weighted_loss)

# Load the saved model
model_path = "saved_model"
loaded_model = TFBertForSequenceClassification.from_pretrained(model_path)
print("Model loaded successfully!")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prediction function
def predict_texts(texts, tokenizer, model, max_length=128):
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Tokenize the inputs
    encodings = tokenizer(
        processed_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="tf"
    )
    
    # Get predictions
    logits = model(encodings).logits
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    predicted_labels = tf.argmax(probabilities, axis=1).numpy()
    
    return processed_texts, predicted_labels, probabilities

# Example usage: Predict single or multiple texts
text_data = pd.read_csv("datasets/test_data.csv")
texts_to_predict = text_data["text"].tolist()

# Get predictions
processed_texts, predictions, probabilities = predict_texts(texts_to_predict, tokenizer, loaded_model)

# Print predictions
label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
for text, processed, label, prob in zip(texts_to_predict, processed_texts, predictions, probabilities):
    print(f"Original Text: {text}")
    print(f"Processed Text: {processed}")
    print(f"Predicted Label: {label_map[label]}")
    print(f"Probabilities: {prob}\n")