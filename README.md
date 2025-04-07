# 🚀 Next Word Predictions using LSTM

## 📌 Project Overview
This project implements **Next Word Prediction** using a **Hybrid Model Approach**, combining:
- **Bidirectional Long Short-Term Memory (BiLSTM)** for predicting known words.
- **Transformer-based Model** for handling out-of-vocabulary (OOV) words.
- **Continuous Learning Mechanism** to improve predictions dynamically.

The system is trained on `metamorphosis_clean.txt` and adapts over time by incorporating new words encountered during predictions.

---

## 🌟 Features
✅ **BiLSTM for In-Vocabulary Words** - Predicts words present in the dataset.
✅ **Transformer Model for OOV Words** - Generates predictions for unknown words.
✅ **Adaptive Learning** - Stores new words and refines predictions dynamically.
✅ **Hybrid Intelligence** - Combines deep learning models for accuracy.
✅ **User Input Storage** - Enhances dataset with user-generated content.
✅ **Language Filtering** - Ensures predictions remain meaningful and pleasant.
✅ **Pre-trained Model** - Enables efficient and fast predictions.

---

## 📁 Project Structure
```
📂 Next-Word-Prediction-LSTM
│── 📜 app.py                        # Flask application for serving predictions
│── 📜 predictions.ipynb              # Jupyter Notebook for generating predictions
│── 📄 next_word_model.h5             # Trained LSTM model file
│── 📄 tokenizer1.pkl                 # Tokenizer used for text processing
│── 📄 metamorphosis_clean.txt        # Dataset for training
│── 📄 index.html                     # Web interface for user input
│── 📖 README.md                      # Project Documentation
```

---

## 🔧 Installation & Setup
1️⃣ Clone this repository:
   ```bash
   git clone https://github.com/your-repo/Next-Word-Prediction-LSTM.git
   cd Next-Word-Prediction-LSTM
   ```
2️⃣ Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Flask application:
   ```bash
   python app.py
   ```
4️⃣ Open `index.html` in a browser to interact with the model.

---

## 🛠 Usage
💡 **Train** the model (if needed) using `predictions.ipynb`.
💡 **Run the Flask app** using `app.py` to serve predictions.
💡 **Use the Web Interface** (`index.html`) to test the model interactively.

---

## 🚀 Future Enhancements
🔹 Optimize hybrid model for **faster** and **more accurate** predictions.
🔹 Expand dataset with **context-aware learning**.
🔹 Enhance the **web UI** for better user experience.

---

## 👥 Contributors
- **[Viswas]** – Developer & Maintainer

📩 *Feel free to contribute or suggest improvements!* ✨

---

## 📜 License
This project is licensed under the **MIT License**.

---
