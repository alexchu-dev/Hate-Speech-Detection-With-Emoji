import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import BertTokenizer, TFBertForSequenceClassification
import seaborn as sns

# from sklearn.utils import resample


# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print(tf.config.list_physical_devices("GPU"))

# Constants
model_name = "bert-base-uncased"
data_file = "datasets/labeled_data_added_emoji.csv"

# Directories for logs, figures, models, and errors
log_dir = "logs"
fig_dir = "figures"
model_dir = "models"
error_analytics_dir = "error_analytics"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(error_analytics_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(
                log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ),
        logging.StreamHandler(),
    ],
)


class HateSpeechDetector:
    def __init__(self, max_length=128, batch_size=32):
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        self.history = None

    def analyse_data_distribution(self, texts, labels):
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

    def preprocess_text(self, text):
        """Preprocess text with emoji handling"""
        import emoji
        import unicodedata2

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
        text = re.sub(r"@\b(?!ss\b|ggot\b|\$\$\b|f@990t\b)\w+", "", text).replace(
            "@", "a"
        )
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

    def prepare_data(self, df, text_column):
        """Prepare and preprocess the dataset."""
        # Log some examples of preprocessed text
        logging.info("\nPreprocessed text examples:")
        for i in range(min(5, len(df))):
            logging.info(f"Original: {df[text_column].iloc[i]}")
            logging.info(
                f"Preprocessed: {self.preprocess_text(df[text_column].iloc[i])}\n"
            )

        # Apply text preprocessing
        df[text_column] = df[text_column].apply(self.preprocess_text)

        # Remove empty strings and reset index
        df = df[df[text_column].str.len() > 0].reset_index(drop=True)

        # Check data distribution
        self.analyse_data_distribution(df[text_column], df["label"])

        return df

    def tokenize_data(self, texts, labels):
        """Tokenize text data for model input."""
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)

        encodings = self.tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
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

    def create_model(self, trainable_layers=2):
        """Initialize the model architecture."""
        if self.model is None:
            self.model = TFBertForSequenceClassification.from_pretrained(
                model_name, num_labels=3
            )

            # Freeze embeddings
            self.model.bert.embeddings.trainable = False

            # Freeze most of the encoder layers
            num_layers = len(self.model.bert.encoder.layer)
            num_layers_to_freeze = num_layers - trainable_layers

            for layer_idx in range(num_layers_to_freeze):
                self.model.bert.encoder.layer[layer_idx].trainable = False

            # Add dropout for regularization
            self.dropout = tf.keras.layers.Dropout(0.1)

            # Log trainable and non-trainable parameters
            trainable_params = sum(
                [tf.keras.backend.count_params(w) for w in self.model.trainable_weights]
            )
            non_trainable_params = sum(
                [
                    tf.keras.backend.count_params(w)
                    for w in self.model.non_trainable_weights
                ]
            )
            logging.info(f"Trainable parameters: {trainable_params:,}")
            logging.info(f"Non-trainable parameters: {non_trainable_params:,}")
            logging.info(
                f"Frozen {num_layers_to_freeze} layers, Trainable {trainable_layers} layers"
            )

        # Create optimizer
        optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=5e-5, decay=0.01, epsilon=1e-6, amsgrad=True
        )

        # Custom loss function with class weights
        def weighted_loss(y_true, y_pred):
            # Calculate base loss
            base_loss = tf.keras.losses.categorical_crossentropy(
                y_true, y_pred, from_logits=True
            )

            # Add class weights (higher weight for hate speech)
            weights = tf.constant([3.4, 1.0, 2.3])
            sample_weights = tf.reduce_sum(y_true * weights, axis=1)

            # Apply weights
            weighted_loss = base_loss * sample_weights
            return tf.reduce_mean(weighted_loss)

        # Compile model with custom loss and metrics
        self.model.compile(
            optimizer=optimizer,
            loss=weighted_loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )

    def train(self, train_df, val_df, text_column, label_column, epochs=11):
        """Train the model with cross-validation."""
        logging.info("Starting model training...")

        # Prepare datasets
        train_df = self.prepare_data(train_df, text_column)
        val_df = self.prepare_data(val_df, text_column)

        # # Apply upsampling
        # train_df = self.upsample_minority(train_df, label_column)

        # Create datasets
        train_ds = self.tokenize_data(train_df[text_column], train_df[label_column])
        val_ds = self.tokenize_data(val_df[text_column], val_df[label_column])

        # Add batching and prefetching
        train_ds = (
            train_ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        )
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        # Create model
        self.create_model()

        # Calculate class weights
        class_weights = self.calculate_class_weights(train_df[label_column])
        filename = os.path.join(
            model_dir, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=filename, monitor="val_loss", save_best_only=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=5e-7
            ),
        ]

        # Train the model
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        logging.info("Model training completed.")
        self.plot_training_history()

    # Upsample the minority classes
    # def upsample_minority(self, df, label_column):
    #     """Upsample the minority class."""
    #     # Separate majority and minority classes
    #     majority_class = df[df[label_column] == 1]
    #     minority_classes = df[df[label_column] != 1]

    #     # Upsample minority class
    #     minority_upsampled = resample(
    #         minority_classes,
    #         replace=True,
    #         n_samples=len(majority_class),
    #         random_state=42,
    #     )

    #     # Combine majority class with upsampled minority class
    #     upsampled_df = pd.concat([majority_class, minority_upsampled])

    #     return upsampled_df.sample(frac=1, random_state=42)

    def calculate_class_weights(self, labels):
        """Calculate balanced class weights."""
        counts = np.bincount(labels)
        total = len(labels)
        weights = {i: total / (len(counts) * count) for i, count in enumerate(counts)}
        return weights

    def plot_training_history(self):
        """Plot training metrics."""
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Training and Validation Metrics\n{data_file}", fontsize=16)

        # Plot loss
        axs[0, 0].plot(self.history.history["loss"], label="Training Loss")
        axs[0, 0].plot(self.history.history["val_loss"], label="Validation Loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()

        # Plot accuracy
        axs[0, 1].plot(self.history.history["accuracy"], label="Training Accuracy")
        axs[0, 1].plot(
            self.history.history["val_accuracy"], label="Validation Accuracy"
        )
        axs[0, 1].set_title("Accuracy")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Accuracy")
        axs[0, 1].legend()

        # Plot precision
        axs[1, 0].plot(self.history.history["precision"], label="Training Precision")
        axs[1, 0].plot(
            self.history.history["val_precision"], label="Validation Precision"
        )
        axs[1, 0].set_title("Precision")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Precision")
        axs[1, 0].legend()

        # Plot recall
        axs[1, 1].plot(self.history.history["recall"], label="Training Recall")
        axs[1, 1].plot(self.history.history["val_recall"], label="Validation Recall")
        axs[1, 1].set_title("Recall")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Recall")
        axs[1, 1].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        filename = os.path.join(
            fig_dir, f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        try:
            plt.savefig(filename)
            logging.info(f"Training history saved successfully: {filename}")
        except Exception as e:
            logging.error(f"Error saving training history: {e}")
        plt.close()

    def evaluate(self, test_df, text_column, label_column):
        """Evaluate the model on test data."""
        logging.info("Starting model evaluation...")

        # Prepare test data
        test_df = self.prepare_data(test_df, text_column)
        test_ds = self.tokenize_data(test_df[text_column], test_df[label_column])
        test_ds = test_ds.batch(self.batch_size)

        # Make predictions
        predictions = []
        true_labels = []
        all_probs = []

        for batch in test_ds:
            logits = self.model(batch[0])[0]
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

        # Print metrics
        for metric, value in metrics.items():
            logging.info(f"{metric.upper()}: {value:.4f}")

        # Create confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        self.plot_confusion_matrix(cm)

        # Error analysis
        self.analyse_errors(test_df[text_column], true_labels, predictions)

        return metrics

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        if cm is None or not isinstance(cm, np.ndarray):
            logging.error("Invalid confusion matrix passed to plot_confusion_matrix")
            return

        print("Plotting confusion matrix...")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix\n{data_file}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        filename = os.path.join(
            fig_dir, f'confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        )
        logging.info(f"Saving confusion matrix to {filename}")
        try:
            plt.savefig(filename)
            logging.info(f"Confusion matrix saved successfully: {filename}")
        except Exception as e:
            logging.error(f"Error saving confusion matrix: {e}")
        plt.close()

    def analyse_errors(self, texts, true_labels, predicted_labels):
        """analyse prediction errors."""
        errors = []
        for text, true, pred in zip(texts, true_labels, predicted_labels):
            if true != pred:
                errors.append(
                    {"text": text, "true_label": true, "predicted_label": pred}
                )

        error_df = pd.DataFrame(errors)
        error_df.to_csv(
            os.path.join(
                error_analytics_dir,
                f'error_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            ),
            index=False,
        )
        logging.info(f"Found {len(errors)} errors. Error analysis saved to CSV.")

    def predict(self, texts):
        """Make predictions on new texts."""
        # Preprocess texts
        if isinstance(texts, str):
            texts = [texts]

        # Create dataset
        dataset = self.tokenize_data(texts, [0] * len(texts))  # Dummy labels
        dataset = dataset.batch(self.batch_size)

        # Make predictions
        predictions = []
        probabilities = []

        for batch in dataset:
            logits = self.model(batch[0])[0]
            probs = tf.nn.softmax(logits, axis=1).numpy()
            preds = tf.argmax(logits, axis=1).numpy()

            predictions.extend(preds)
            probabilities.extend(probs)

        return predictions, probabilities

    def save_model(self, path):
        """Save the model to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logging.info(f"Model and tokenizer saved to {path}")

    @classmethod
    def load_model(cls, path, max_length=128, batch_size=32):
        """Load the model from the specified path."""
        detector = cls(max_length=max_length, batch_size=batch_size)
        detector.model = TFBertForSequenceClassification.from_pretrained(path)
        detector.tokenizer = BertTokenizer.from_pretrained(path)
        logging.info(f"Model and tokenizer loaded from {path}")
        return detector


if __name__ == "__main__":
    # Load your data
    df = pd.read_csv(data_file)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    # Initialize detector
    detector = HateSpeechDetector(batch_size=128)

    # Train model
    detector.train(
        train_df=train_df,
        val_df=val_df,
        text_column="text",
        label_column="label",
        epochs=20,
    )

    # Save model
    detector.save_model("best_model")

    # Evaluate model
    metrics = detector.evaluate(
        test_df=test_df, text_column="text", label_column="label"
    )

    print("Evaluation Metrics:", metrics)

    # Test the saved model
    print("Testing the saved model...")

    # Load the saved model
    detector = HateSpeechDetector.load_model("best_model")
    # Load test data
    test_texts = pd.read_csv("datasets/test_data_added_emoji.csv")["text"]
    test_texts_original_labels = pd.read_csv("datasets/test_data_added_emoji.csv")[
        "label"
    ]
    processed_test_texts = [detector.preprocess_text(t) for t in test_texts]
    predictions, probabilities = detector.predict(processed_test_texts)

    # Print predictions
    label_map = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral"}
    for text, original_label, processed_test_text, label, prob in zip(
        test_texts,
        test_texts_original_labels,
        processed_test_texts,
        predictions,
        probabilities,
    ):
        print(f"Original Text: {text}")
        print(f"Processed text: {processed_test_text}")
        print(f"Original Label: {label_map[original_label]}")
        print(f"Predicted Label: {label_map[label]}")
        print(f"Probabilities: {prob}\n")
