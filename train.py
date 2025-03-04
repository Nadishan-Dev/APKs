import tensorflow as tf
import pandas as pd
import numpy as np
import json
import pickle
import os
import datetime
import shutil
import sys  # Added for clean exit
import argparse  # Added for command line arguments
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, \
    concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Create directories if they don't exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../datasets", exist_ok=True)
os.makedirs("../models/checkpoints", exist_ok=True)
os.makedirs("../models/history", exist_ok=True)


# Load Dataset from JSON with improved parsing for question-answer pairs
def load_data_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize lists to store extracted data
    inputs = []
    responses = []

    # Process each item in the JSON data
    for item in data:
        # Process each QA pair in the item
        for qa_pair in item.get('qa_pairs', []):
            # Add the main question and answer pair
            question = qa_pair.get('question', '')
            answer = qa_pair.get('answer', '')

            if question and answer:
                inputs.append(question)
                responses.append(answer)

            # Process question variations for the current question
            variations = qa_pair.get('question_variations', [])
            for variation in variations:
                if variation:
                    inputs.append(variation)
                    responses.append(answer)  # Same answer for all variations

    # Create DataFrame from the extracted data
    df = pd.DataFrame({'input': inputs, 'response': responses})
    return df


# Create a function to collect new training data from user interactions
def save_new_training_data(question, correct_answer, dataset_path="../datasets/user_feedback.json"):
    """
    Save user feedback as new training data
    """
    new_data = {
        "question": question,
        "answer": correct_answer
    }

    # Create or append to user feedback file
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(new_data)

    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"New training data saved to {dataset_path}")


# Add enhanced data augmentation
def augment_data(df, augmentation_factor=2):
    """
    More robust data augmentation with multiple techniques
    """
    augmented_data = df.copy()
    augmented_samples = []

    # Word substitution dictionary could be expanded
    word_substitutions = {
        # Example English substitutions - replace with relevant ones for your use case
        "how": ["what way", "in what manner"],
        "what": ["which", "what kind of"],
        "where": ["in which place", "at what location"],
        "when": ["at what time", "on which date"],
        "who": ["which person", "what person"],
        "why": ["for what reason", "how come"],
        "you": ["yourself", "DORA"],
        "hello": ["hi", "greetings", "hey"],
        "thanks": ["thank you", "appreciated", "grateful"],
        "help": ["assist", "aid", "support"]
    }

    for idx, row in df.iterrows():
        text = row['input']
        words = text.split()

        # Skip very short inputs for augmentation
        if len(words) <= 2:
            continue

        # Create variations with word substitutions
        for _ in range(augmentation_factor):
            if len(words) > 2:
                # Random word substitution
                num_replacements = np.random.randint(1, min(2, len(words)))
                indices = np.random.choice(len(words), num_replacements, replace=False)

                new_words = words.copy()
                substitution_made = False

                for idx in indices:
                    if words[idx].lower() in word_substitutions:
                        new_words[idx] = np.random.choice(word_substitutions[words[idx].lower()])
                        substitution_made = True

                # Only add if we actually made a change
                if substitution_made:
                    new_text = ' '.join(new_words)
                    augmented_samples.append({
                        'input': new_text,
                        'response': row['response']
                    })

        # Word order shuffling (for some languages this works better than others)
        if len(words) > 3:
            # Keep first and last word, shuffle middle words
            middle_words = words[1:-1]
            if len(middle_words) > 1:
                np.random.shuffle(middle_words)
                new_text = words[0] + ' ' + ' '.join(middle_words) + ' ' + words[-1]
                augmented_samples.append({
                    'input': new_text,
                    'response': row['response']
                })

    if augmented_samples:
        augmented_df = pd.DataFrame(augmented_samples)
        augmented_data = pd.concat([augmented_data, augmented_df], ignore_index=True)
        print(f"Augmented data from {len(df)} to {len(augmented_data)} examples")

    return augmented_data


# Function to merge existing and new datasets
def merge_datasets(main_dataset_path, new_data_path="../datasets/user_feedback.json"):
    """
    Merge existing training data with new user feedback data
    """
    # Load main dataset
    main_df = load_data_from_json(main_dataset_path)

    # Check if new data exists
    if os.path.exists(new_data_path):
        try:
            # Convert user feedback format to the main dataset format
            with open(new_data_path, 'r', encoding='utf-8') as f:
                new_data = json.load(f)

            # Create a new JSON structure matching the main dataset format
            formatted_data = [{
                "qa_pairs": []
            }]

            for item in new_data:
                qa_pair = {
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_variations": []  # Could be populated in the future
                }
                formatted_data[0]["qa_pairs"].append(qa_pair)

            # Save the formatted data temporarily
            temp_file = "../datasets/temp_formatted.json"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, ensure_ascii=False)

            # Load the formatted data
            new_df = load_data_from_json(temp_file)

            # Remove temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

            # Check if there's new data
            if len(new_df) > 0:
                print(f"Found {len(new_df)} new training examples from user feedback")

                # Combine datasets
                combined_df = pd.concat([main_df, new_df], ignore_index=True)

                # Remove duplicates based on input
                combined_df.drop_duplicates(subset=['input'], keep='last', inplace=True)

                print(f"Combined dataset has {len(combined_df)} examples")
                return combined_df
        except Exception as e:
            print(f"Error merging datasets: {str(e)}")

    return main_df


# Build an improved model with additional capabilities
def build_qa_model(vocab_size, embedding_dim, max_length, num_classes, dropout_rate=0.5):
    """
    Build an improved deep learning model for question answering
    """
    # Input layer
    input_layer = Input(shape=(max_length,))

    # Embedding layer with more dimensions
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        mask_zero=True
    )(input_layer)

    # Deeper LSTM architecture
    lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
    dropout_1 = Dropout(dropout_rate / 2)(lstm_1)

    lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(dropout_1)
    dropout_2 = Dropout(dropout_rate / 2)(lstm_2)

    lstm_3 = Bidirectional(LSTM(32, return_sequences=True))(dropout_2)

    # Multiple pooling strategies for better feature extraction
    max_pooling = GlobalMaxPooling1D()(lstm_3)
    avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(lstm_3)

    # Concatenate pooled features
    concatenated = concatenate([max_pooling, avg_pooling])

    # Deeper and wider dense layers
    dense1 = Dense(256, activation="relu")(concatenated)
    dropout1 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(128, activation="relu")(dropout1)
    dropout2 = Dropout(dropout_rate / 2)(dense2)

    dense3 = Dense(64, activation="relu")(dropout2)

    # Output layer
    output_layer = Dense(num_classes, activation="softmax")(dense3)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# Function to load or create a model
def get_model(vocab_size, embedding_dim, max_length, num_classes, model_path="../models/dora_01.h5"):
    """
    Load existing model or create a new one if it doesn't exist
    """
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        try:
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Building new model instead...")

    print("Building new model...")
    model = build_qa_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        max_length=max_length,
        num_classes=num_classes
    )

    # Compile model with learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


# Function to predict answer with confidence threshold and fallback
def predict_answer(question, model, tokenizer, label_encoder, max_length=50, confidence_threshold=0.5):
    """
    Predict answer with confidence check and fallback for low-confidence predictions
    """
    # Preprocess the question
    sequence = tokenizer.texts_to_sequences([question])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict the class
    prediction = model.predict(padded)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    # Get the answer
    answer = label_encoder.inverse_transform([predicted_class])[0]

    # Return with confidence information
    return {
        'answer': answer,
        'confidence': float(confidence),
        'is_confident': confidence >= confidence_threshold,
        'top_classes': np.argsort(-prediction)[:3],  # Get top 3 predictions for debugging
        'top_confidences': -np.sort(-prediction)[:3]  # Get confidences for top 3
    }


# Function to retrain model with new data
def retrain_model(model, current_epoch=0, max_epochs=20):
    """
    Retrain model with combined data (original + user feedback)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create checkpoint for current model
    checkpoint_path = f"../models/checkpoints/dora_01_{timestamp}.h5"
    model.save(checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Load and merge datasets
    main_dataset_path = "../datasets/DORA_ds_01.json"
    df = merge_datasets(main_dataset_path)

    # Apply data augmentation
    df = augment_data(df)

    # Tokenization & Encoding
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["input"])
    sequences = tokenizer.texts_to_sequences(df["input"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Save updated tokenizer
    with open(f"../models/tokenizer_{timestamp}.json", "w", encoding="utf-8") as f:
        tokenizer_json = tokenizer.to_json()
        json.dump(json.loads(tokenizer_json), f, ensure_ascii=False)

    # Make a copy of the current tokenizer as the main one
    shutil.copy(f"../models/tokenizer_{timestamp}.json", "../models/tokenizer.json")

    # Update label encoding
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["response"])
    num_classes = len(label_encoder.classes_)

    # Save updated label encoder
    with open(f"../models/label_encoder_{timestamp}.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Make a copy of the current label encoder as the main one
    shutil.copy(f"../models/label_encoder_{timestamp}.pkl", "../models/label_encoder.pkl")

    # Save label mapping as JSON
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open(f"../models/label_encoder_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False)

    # Make a copy of the current label mapping as the main one
    shutil.copy(f"../models/label_encoder_{timestamp}.json", "../models/label_encoder.json")

    # Split data for training/validation
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )

    # Adaptive learning rate based on current epoch
    current_lr = LEARNING_RATE * (0.9 ** (current_epoch // 5))  # Reduce LR every 5 epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=current_lr)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"../models/dora_01_best_{timestamp}.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
    ]

    # Train for fewer epochs when retraining
    epochs = min(max_epochs, EPOCHS // 2)

    # Train model
    print(f"Retraining model with {len(df)} examples for {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # Save updated model
    model.save(f"../models/dora_01_{timestamp}.h5")
    model.save("../models/dora_01.h5")  # Replace main model

    # Save training history
    history_path = f"../models/history/training_history_{timestamp}.json"
    with open(history_path, "w") as f:
        history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
        json.dump(history_dict, f)

    # Update the current model config
    model_config = {
        "model_name": "dora_01",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1),
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": num_classes,
        "version": "3.0",
        "last_retrained": timestamp,
        "total_training_examples": len(df),
        "data_format": "JSON",
        "checkpoint_path": checkpoint_path
    }

    with open("../models/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print(f"Model retrained and saved. New version: dora_01_{timestamp}.h5")

    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    return model, tokenizer, label_encoder


# Define hyperparameters
MAX_VOCAB_SIZE = 10000  # Increased vocabulary size
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 200  # Increased embedding dimensions
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
RETRAINING_THRESHOLD = 50  # Number of new examples before retraining


# Main function to train or load the model
def main(retrain=False, auto_exit=False):
    # Load data
    json_file_path = "../datasets/DORA_ds_01.json"
    df = load_data_from_json(json_file_path)
    print(f"Dataset loaded with {len(df)} examples")

    # Apply data augmentation if needed
    df = augment_data(df)

    # Preprocessing
    print("Preprocessing data...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["input"])
    sequences = tokenizer.texts_to_sequences(df["input"])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Save vocabulary
    word_index = tokenizer.word_index
    vocab_size = min(MAX_VOCAB_SIZE, len(word_index) + 1)
    print(f"Vocabulary size: {vocab_size}")

    with open("../models/vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False)

    # Label encoding
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df["response"])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Save label mapping
    label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open("../models/label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False)

    # Save LabelEncoder
    with open("../models/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )

    # Get or create model
    model = get_model(vocab_size, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, num_classes)

    # Display model summary
    model.summary()

    # Retrain if requested or if new data available
    if retrain or (os.path.exists("../datasets/user_feedback.json") and
                   len(json.load(open("../datasets/user_feedback.json", 'r'))) >= RETRAINING_THRESHOLD):
        model, tokenizer, label_encoder = retrain_model(model)
    else:
        # Train model if it's new
        if not os.path.exists("../models/dora_01.h5"):
            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath="../models/dora_01_best.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]

            # Train model
            print("Training model...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks
            )

            # Save model and history
            model.save("../models/dora_01.h5")

            with open("../models/tokenizer.json", "w", encoding="utf-8") as f:
                tokenizer_json = tokenizer.to_json()
                json.dump(json.loads(tokenizer_json), f, ensure_ascii=False)

            with open("../models/training_history.json", "w") as f:
                history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
                json.dump(history_dict, f)

    # Save model configuration
    model_config = {
        "model_name": "dora_01",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "num_classes": num_classes,
        "version": "3.0",
        "data_format": "JSON",
        "last_updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open("../models/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    print("Model ready for use!")

    # Test the model with sample questions
    test_model(model, tokenizer, label_encoder, df)

    # If auto_exit is True, return from main without entering interactive mode
    if auto_exit:
        print("\nTraining and testing complete. Exiting as requested.")
        return model, tokenizer, label_encoder

    return model, tokenizer, label_encoder


# Function to test model with sample questions
def test_model(model, tokenizer, label_encoder, df=None):
    print("\nTesting model with sample questions:")

    if df is not None and len(df) > 0:
        # Test with examples from dataset
        for i in range(min(5, len(df))):
            test_question = df['input'].iloc[i]
            actual_answer = df['response'].iloc[i]

            result = predict_answer(
                test_question,
                model,
                tokenizer,
                label_encoder
            )

            print(f"\nQuestion: {test_question}")
            print(f"Predicted answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Actual answer: {actual_answer}")
            print(f"Correct: {'✓' if result['answer'] == actual_answer else '✗'}")

    # Test with custom examples
    examples = [
        "Who are you?",
        "What can you help me with?",
        "Tell me about yourself."
    ]

    for example in examples:
        result = predict_answer(
            example,
            model,
            tokenizer,
            label_encoder
        )

        print(f"\nExample Question: {example}")
        print(f"Predicted answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Is confident: {'Yes' if result['is_confident'] else 'No'}")


# Simple interactive mode for testing and collecting feedback
def interactive_mode():
    """
    Interactive session for testing the model and collecting user feedback
    """
    print("\n====== DORA_01 Interactive Mode ======")
    print("Ask questions and provide feedback to help DORA learn.")
    print("Type 'exit' to quit, or 'retrain' to force model retraining.")

    # Load model and required components
    try:
        model = load_model("../models/dora_01.h5")

        with open("../models/tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_json = json.load(f)
            tokenizer = Tokenizer()
            tokenizer._token_counts = json.loads(tokenizer_json["word_counts"])
            tokenizer.word_index = json.loads(tokenizer_json["word_index"])
            tokenizer.index_word = json.loads(tokenizer_json["index_word"])
            tokenizer.num_words = tokenizer_json["num_words"]

        with open("../models/label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Please train the model first by running the main() function.")
        return

    # Track feedback for retraining decision
    feedback_count = 0

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == 'retrain':
            print("Retraining model with collected feedback...")
            model, tokenizer, label_encoder = retrain_model(model)
            feedback_count = 0
            continue

        if not user_input:
            continue

        # Get prediction
        result = predict_answer(
            user_input,
            model,
            tokenizer,
            label_encoder
        )

        print(f"DORA: {result['answer']}")
        print(f"(Confidence: {result['confidence']:.2f})")

        # Ask for feedback if confidence is low or randomly (20% chance)
        if not result['is_confident'] or np.random.random() < 0.2:
            feedback = input("Is this answer correct? (y/n): ").strip().lower()

            if feedback == 'n':
                correct_answer = input("What would be the correct answer? ").strip()

                if correct_answer:
                    # Save feedback for future training
                    save_new_training_data(user_input, correct_answer)
                    feedback_count += 1

                    print(
                        f"Thank you! Your feedback has been saved. ({feedback_count}/{RETRAINING_THRESHOLD} until auto-retraining)")

                    # Retrain if enough feedback collected
                    if feedback_count >= RETRAINING_THRESHOLD:
                        print("\nCollected enough feedback. Retraining model...")
                        model, tokenizer, label_encoder = retrain_model(model)
                        feedback_count = 0


# Script execution
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='DORA: Deep Learning Q&A Model')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model with new data')
    parser.add_argument('--interactive', action='store_true', help='Start interactive mode after training')
    parser.add_argument('--auto-exit', action='store_true', help='Exit after training and testing')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        # Default behavior: show help and exit
        parser.print_help()
        print("\nNo arguments provided. Using default behavior (train and start interactive mode).")
        # Run main to train/load the model
        model, tokenizer, label_encoder = main(retrain=False, auto_exit=False)
        # Start interactive mode
        interactive_mode()
    else:
        # Run main with specified options
        model, tokenizer, label_encoder = main(retrain=args.retrain, auto_exit=args.auto_exit)

        # Only start interactive mode if requested and not auto-exiting
        if args.interactive and not args.auto_exit:
            interactive_mode()
        elif not args.auto_exit:
            print("\nTraining and testing complete.")
            print("To start interactive mode, run the script with --interactive flag.")
            print("Exiting...")