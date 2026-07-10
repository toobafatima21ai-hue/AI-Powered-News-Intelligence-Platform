# 📰 AI-Powered News Topic Classifier using BERT

An end-to-end NLP classification system that fine-tunes a **BERT Transformer model** to automatically categorize news headlines into different topics. The project includes model training, evaluation, explainable predictions, confidence analysis, and an interactive Streamlit deployment dashboard.

---

## 🚀 Project Overview

This project implements a **Transformer-based News Classification System** using **BERT (Bidirectional Encoder Representations from Transformers)** and Hugging Face Transformers.

The model is trained on the **AG News Dataset** to classify news headlines into four categories:

- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Sci/Tech

The application provides real-time predictions with confidence scores and additional AI-powered insights through an interactive Streamlit interface.

---

# ✨ Features

## 🤖 BERT-Based Text Classification
- Fine-tuned `bert-base-uncased` Transformer model
- Transfer learning approach for NLP classification
- Automatic headline topic prediction

## 📊 Confidence Analysis
- Displays prediction confidence score
- Visualizes probability distribution across all categories
- Identifies uncertain predictions

## 🔍 Explainable AI Insights
- Extracts important keywords from headlines
- Provides related category keywords
- Improves model interpretability

## 📚 Batch Prediction
- Classify multiple headlines simultaneously
- Export and analyze multiple predictions efficiently

## 📈 Analytics Dashboard
- Dataset category distribution visualization
- Prediction history tracking
- Interactive charts using Plotly

## 🌐 Streamlit Deployment
- User-friendly web interface
- Real-time AI predictions
- Portfolio-ready application design

---

# 🏗️ Project Architecture

```
News_Topic_Classifier/

│
├── train.py
│   └── BERT model fine-tuning pipeline
│
├── app.py
│   └── Streamlit interactive application
│
├── train.csv
├── test.csv
│   └── AG News dataset
│
├── bert_news_classifier/
│   └── Saved fine-tuned BERT model
│
└── README.md
```

---

# 🛠️ Technologies Used

### Programming Language
- Python

### Natural Language Processing
- Hugging Face Transformers
- BERT
- Tokenization
- Transfer Learning

### Deep Learning
- PyTorch

### Data Processing
- Pandas
- NumPy

### Model Deployment
- Streamlit

### Visualization
- Plotly
- Matplotlib

### Explainable AI
- YAKE Keyword Extraction

---

# 📂 Dataset

## AG News Dataset

The model is trained using the **AG News Dataset**, a benchmark dataset containing news headlines categorized into four classes.

Dataset Categories:

| Label | Category |
|------|----------|
| 1 | World |
| 2 | Sports |
| 3 | Business |
| 4 | Sci/Tech |

Dataset contains:

- 120,000 training samples
- 7,600 testing samples

---

# ⚙️ Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/News_Topic_Classifier.git

cd News_Topic_Classifier
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Model Training

Run:

```bash
python train.py
```

The training pipeline performs:

1. Dataset loading
2. Text preprocessing
3. BERT tokenization
4. Model fine-tuning
5. Evaluation
6. Saving trained model

The trained model is stored in:

```
bert_news_classifier/
```

---

# 🌐 Run Streamlit Application

Start the application:

```bash
streamlit run app.py
```

The application provides:

- News headline classification
- Confidence scores
- Keyword explanations
- Batch predictions
- Analytics dashboard

---

# 📸 Application Workflow

```
User Input
     |
     ↓
News Headline
     |
     ↓
BERT Tokenizer
     |
     ↓
Fine-Tuned BERT Model
     |
     ↓
Topic Prediction
     |
     ↓
Confidence + Explanation
```

---

# 📊 Model Evaluation

Evaluation Metrics:

- Accuracy
- F1 Score

The model performance is evaluated on unseen test data to measure generalization ability.

---

# 🔮 Future Improvements

- Add multilingual news classification
- Integrate larger transformer models (RoBERTa, DeBERTa)
- Add sentiment analysis
- Deploy using Docker and cloud platforms
- Add continuous learning from user feedback

---

# 🎯 Skills Demonstrated

- Natural Language Processing
- Transformer Models
- BERT Fine-Tuning
- Transfer Learning
- Deep Learning
- Text Classification
- Explainable AI
- Model Deployment
- Streamlit Application Development

---

# 👩‍💻 Author

**Tooba Fatima**

AI/ML Engineer | Generative AI | NLP | Deep Learning

GitHub:
https://github.com/toobafatima21ai-hue

 
