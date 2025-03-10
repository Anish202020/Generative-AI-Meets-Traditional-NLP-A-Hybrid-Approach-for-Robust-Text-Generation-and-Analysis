# 🌟 Generative AI Meets Traditional NLP: A Hybrid Approach for Robust Text Generation and Analysis

   

## 🚀 Overview

This project integrates Generative AI and Traditional NLP to create a hybrid approach for text generation and analysis. The system:

📌 Generates text using GPT-2 (a transformer-based language model).

📌 Performs linguistic analysis (Named Entity Recognition & Part-of-Speech tagging) using spaCy.

📌 Conducts sentiment analysis using TextBlob.

## ✨ Features

✅ Text Generation – Uses GPT-2 to generate contextual text based on a prompt.✅ Named Entity Recognition (NER) – Extracts entities like names, places, and organizations.✅ Part-of-Speech (POS) Tagging – Identifies grammatical roles (e.g., noun, verb, adjective).✅ Sentiment Analysis – Determines the polarity (positive/negative/neutral) and subjectivity of text.

🛠️ Tech Stack

Python 3.8+

spaCy (en_core_web_sm model)

Hugging Face Transformers (GPT-2)

Torch (PyTorch)

TextBlob

## 📂 Installation

First, clone the repository:

 git clone https://github.com/your-username/Generative-AI-Meets-Traditional-NLP.git
 cd Generative-AI-Meets-Traditional-NLP

Then install dependencies:

pip install -r requirements.txt

## 🚀 How to Run

Run the script:

python main.py

Example output:

Generated Text:
"In the future, artificial intelligence will revolutionize how we interact with technology."

Analyzing Generated Text:
Named Entities:
(None, since GPT-2 text might not contain named entities)

Part-of-Speech Tags:
future - NOUN
artificial - ADJ
intelligence - NOUN

Sentiment Analysis:
Polarity: 0.3, Subjectivity: 0.5

## 📌 Code Structure

### 📂 Generative-AI-Meets-Traditional-NLP
 ├── 📜 main.py  # Entry point
 ├── 📜 requirements.txt  # Dependencies
 ├── 📜 README.md  # Documentation




👨‍💻 Author

Developed by [Anish Kumar] – Feel free to reach out! 🚀

🌟 If you like this project, consider giving it a ⭐ on GitHub!
