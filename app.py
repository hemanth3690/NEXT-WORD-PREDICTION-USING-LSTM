from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
from transformers import pipeline
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from keras.models import load_model  # type: ignore

app = Flask(__name__)

# Paths
MODEL_PATH = os.path.abspath("bilstm_model.h5")
TOKENIZER_PATH = os.path.abspath("tokenizer1.pkl")
DATASET_PATH = os.path.abspath("metamorphosis_clean.txt")

# Load BiLSTM Model
def load_bilstm_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, compile=False)
            print("✅ BiLSTM model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Error loading BiLSTM model: {e}")
    else:
        print(f"❌ Model file not found at {MODEL_PATH}.")
    return None

bilstm_model = load_bilstm_model()

# Load tokenizer
try:
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Error loading tokenizer: {e}")
    tokenizer = None

# Load BERT Model (fill-mask pipeline)
fill_mask = pipeline("fill-mask", model="bert-base-uncased")

# Function to store only the predicted sentence in a .txt file
# -----------------------------------------------------------
def store_user_data(predicted_text, filename='metamorphosis_clean.txt'):
    """
    Append only the predicted text to a text file.
    Each prediction is stored on a new line.
    """
    with open(filename, "a", encoding="utf-8") as f:
        f.write(predicted_text + "\n")

        
# Load dataset words
dataset_words = set()
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset_words.update(f.read().split())

# Inappropriate words filter
BAD_WORDS = {"damn", "hell", "shit", "fuck", "bitch", "bastard", "asshole", "dumbass", "jackass", 
             "motherfucker", "cock", "piss", "crap", "slut", "whore", "dick", "cunt", "nigger", 
             "retard", "faggot", "twat", "wanker", "moron", "idiot", "stupid"}

def is_valid_word(word):
    return word.lower() not in BAD_WORDS

# Predict using BiLSTM with top-k filtering
def predict_next_word_bilstm(text, top_k=5):
    if bilstm_model is None or tokenizer is None:
        return "Error: Model or tokenizer not loaded"
    sequence = tokenizer.texts_to_sequences([text])
    # Adjust the maxlen to match your model's training config. For example, if you trained with 10 tokens:
    padded_sequence = pad_sequences(sequence, maxlen=10, padding='pre')
    predictions = bilstm_model.predict(padded_sequence)[0]
    sorted_indices = np.argsort(predictions)[::-1]
    
    for idx in sorted_indices[:top_k]:
        predicted_word = tokenizer.index_word.get(idx, None)
        if predicted_word and is_valid_word(predicted_word):
            return predicted_word
    return "[filtered]"

# Predict using BERT (fill-mask) and remove subword markers
def predict_next_word_bert(text):
    masked_text = text + " [MASK]."
    predictions = fill_mask(masked_text)
    for pred in predictions:
        word = pred['token_str'].strip()
        if word.startswith("##"):
            word = word[2:]
        if is_valid_word(word):
            return word
    return "[filtered]"

# Hybrid Prediction: check against tokenizer's vocabulary instead of dataset_words
def predict_next_word(text):
    words = text.split()
    last_word = words[-1] if words else ""
    # Check if the last word exists in the tokenizer's vocabulary
    if last_word.lower() in tokenizer.word_index:
        return predict_next_word_bilstm(text)
    else:
        new_word = predict_next_word_bert(text)
        if new_word != "[filtered]":
            dataset_words.add(new_word)
            # Append new word to dataset file with a newline
            with open(DATASET_PATH, 'a', encoding='utf-8') as f:
                f.write("\n" + new_word)
        return new_word

# Predict multiple words
def Predict_Next_Words(text, num_words):
    predicted_sentence = text
    for _ in range(num_words):
        next_word = predict_next_word(predicted_sentence)
        predicted_sentence += " " + next_word.strip()
    return predicted_sentence

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html', text="", predicted_sentence="", error="")

@app.route('/predict', methods=['POST'])
def predict_view():
    text = request.form.get('text', '').strip()
    try:
        num_words = int(request.form.get('num_words', 1))
    except ValueError:
        return render_template('index.html', text=text, error="Invalid word count")
    if bilstm_model is None or tokenizer is None:
        return render_template('index.html', text=text, error="Model or tokenizer not loaded")
    predicted_sentence = Predict_Next_Words(text, num_words)
    store_user_data(predicted_sentence)  # Save the predicted sentence
    return render_template('index.html', text=text, predicted_sentence=predicted_sentence)

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json()
    text = data.get('text', '').strip()
    try:
        num_words = int(data.get('num_words', 1))
    except ValueError:
        return jsonify({'error': "Invalid word count"}), 400
    if bilstm_model is None or tokenizer is None:
        return jsonify({'error': "Model or tokenizer not loaded"}), 400
    predicted_sentence = Predict_Next_Words(text, num_words)  # Generate the predicted sentence
    store_user_data(predicted_sentence)  # Save the predicted sentence
    return jsonify({'predicted_sentence': predicted_sentence})
    return jsonify({'predicted_sentence': predicted_sentence})

if __name__ == '__main__':
    app.run(debug=True)
