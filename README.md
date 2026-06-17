# 💻 AI Laptop Recommender System

An intelligent laptop recommendation web app powered by NLP search, IR ranking, and a machine learning model — built with Python and Flask.

## 📖 About

Users describe what they need in plain English ("a laptop for gaming under $1000") and the system finds the best matches using natural language processing, query expansion, and a trained recommendation model.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **ML / AI:** scikit-learn, sentence-transformers, NLTK
- **Data:** Laptops dataset (CSV/JSON)
- **Search:** TF-IDF IR search + NLP semantic search

## ✨ Features

- 🔍 **NLP Search** — natural language query understanding
- 🧠 **Query Parser** — spec extraction (RAM, GPU, price, brand)
- 📊 **IR Search** — TF-IDF ranked retrieval
- 🤖 **AI Recommender** — trained ML model for personalized results
- 📈 **EDA Dashboard** — exploratory data analysis stats
- 🌐 **Web Interface** — clean Flask app with search and detail pages

## 🚀 Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the app
```bash
python app.py
```
Then open `http://localhost:5000` in your browser.

## 📁 Project Structure

```
laptop-recommender-system/
├── app.py              # Flask web application & routes
├── model.py            # ML recommendation model
├── nlp_search.py       # NLP-based semantic search
├── ir_search.py        # TF-IDF information retrieval
├── query_parser.py     # Query normalization & spec extraction
├── eda.py              # Exploratory data analysis
├── requirements.txt    # Python dependencies
└── data/
    └── laptops.csv     # Laptop dataset
```

## 🎓 Course

University Information Retrieval / AI Course — Python Project
