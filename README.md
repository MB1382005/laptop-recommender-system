# 💻 AI Laptop Recommender System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-powered laptop recommendation system with NLP search, intent detection, and a full Flask web interface**  
*Supports Arabic & English queries — find the right laptop by describing your needs in natural language*

[✨ Features](#-features) • [🧠 How It Works](#-how-it-works) • [🚀 Quick Start](#-quick-start) • [📁 Structure](#-project-structure)

</div>

---

## ✨ Features

- 🔍 **Smart NLP Search** — understands natural language in both **Arabic & English**
- 🎯 **Intent Detection** — recognizes use cases: gaming, programming, video editing, ML/AI, office work, music production, architecture/AutoCAD, portable/travel
- 🤖 **AI Recommender** — cosine-similarity model built with TensorFlow and scikit-learn
- 🔎 **IR Fallback Search** — BM25-style keyword search for broad queries
- 📊 **EDA Dashboard** — built-in analytics with interactive Chart.js visualizations
- 🌐 **Flask Web App** — clean dark-themed UI with search filters, product cards, and detail pages
- 🧩 **Query Parser** — handles GPU/CPU shorthand (`RTX4060` → `RTX 4060`), Arabic specs (`16جيجا` → `16 GB`)

---

## 🧠 How It Works

### 1. Query Processing Pipeline

```
User Query (AR/EN)
      ↓
  Query Parser
  • Normalize GPU/CPU patterns  (RTX4060 → RTX 4060)
  • Translate Arabic specs       (16جيجا → 16 GB)
  • Translate Arabic use-cases   (مونتاج → video editing)
      ↓
  Intent Detection (NLP)
  • Detects: gaming, ML/AI, programming, video_editing,
             office, architecture, music_production, portable
  • Supports multi-intent (e.g. "gaming + lightweight")
      ↓
  Hard Filter → Scoring → Ranked Results
```

### 2. Intent → Scoring Weights

| Intent | GPU | CPU | RAM | SSD | Screen |
|---|---|---|---|---|---|
| **Gaming** | 20 | 8 | 6 | 3 | 3 |
| **Machine Learning / AI** | 10 | 12 | 10 | 5 | 1 |
| **Video Editing** | 8 | 15 | 10 | 6 | 5 |
| **Programming** | 2 | 12 | 8 | 5 | 2 |
| **Portable / Travel** | 1 | 5 | 4 | 3 | 2 |

### 3. AI Recommendation Model

The `model.py` AI recommender uses a **TensorFlow neural embedding model + cosine similarity**:

```python
# Architecture
Input (laptop feature vector)
  → Dense(128, relu)
  → Dense(64, relu)
  → L2-normalized embedding
  → Cosine Similarity → Top-K recommendations
```

Built with `scikit-learn` preprocessing (LabelEncoder, MinMaxScaler) and saved to `model_artifacts/ai_recommender.pkl`.

---

## 🌐 Web Interface

The Flask app (`app.py`) provides a full dark-themed web experience:

| Page | Route | Description |
|---|---|---|
| **Home / Search** | `/` | NLP search + filter by brand, GPU, RAM, price |
| **Product Detail** | `/laptop/<id>` | Full specs + "You may also like" AI recommendations |
| **EDA Dashboard** | `/eda` | Charts: brand distribution, price vs RAM, GPU breakdown |

### Search Examples

```
# English
"RTX 4060 gaming laptop under 1500"
"lightweight laptop for university"
"best laptop for deep learning"

# Arabic
"لاب توب جيمنج خفيف مع RTX"
"لاب توب مونتاج فيديو"
"لاب توب للجامعة رخيص"
"لاب توب برمجة وذكاء اصطناعي"
```

---

## 📁 Project Structure

```
laptop-recommender-system/
│
├── app.py                   # Flask web application & all routes
├── model.py                 # TF embedding model + cosine similarity recommender
├── nlp_search.py            # Intent detection & weighted scoring engine
├── ir_search.py             # BM25-style keyword / IR fallback search
├── query_parser.py          # AR/EN query normalization & expansion
├── eda.py                   # EDA statistics (brand dist, price, GPU breakdown)
│
├── data/
│   └── laptops.csv          # Laptop dataset
├── model_artifacts/
│   └── ai_recommender.pkl   # Pre-trained recommendation model
│
├── laptops.csv              # Dataset (root copy)
├── laptops.json             # JSON version of dataset
└── output__4_.csv           # Processed/output data
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install flask scikit-learn tensorflow pandas numpy joblib datasets
```

### Run Locally

```bash
# Clone the repository
git clone https://github.com/MB1382005/laptop-recommender-system.git
cd laptop-recommender-system

# Run the Flask app
python app.py
```

Then open your browser at **http://localhost:5000**

---

## 🔍 Module Breakdown

### `nlp_search.py` — Smart Search Engine

- Detects **8+ use-case intents** (gaming, ML, video editing, office, architecture, music, portable, programming)
- Supports **full Arabic keyword dictionary** mapped to English specs
- Multi-intent detection — "gaming + lightweight" triggers both intent weight profiles
- Returns `score_breakdown` per laptop for transparent ranking

### `query_parser.py` — Query Normalizer

- Fixes GPU shorthand: `RTX4060` → `RTX 4060`, `RX6700M` → `RX 6700M`
- Fixes CPU shorthand: `ryzen7` → `ryzen 7`, `corei7` → `core i7`
- Translates Arabic specs: `16جيجا` → `16 GB`, `1تيرا` → `1 TB`
- Translates Arabic use-cases: `مونتاج` → `video editing`, `جيمنج` → `gaming`
- Query expansion: adds synonyms for better recall

### `ir_search.py` — IR Fallback

- Keyword-based search over laptop specs
- Used as fallback when NLP confidence is low

### `eda.py` — Analytics Module

- Brand distribution, average price by brand
- RAM size distribution, storage breakdown
- GPU category analysis
- Price correlation with specs

---

## 🔮 Future Improvements

- [ ] **User accounts** — save favorites and search history
- [ ] **Price tracking** — alert when a laptop drops in price
- [ ] **LLM integration** — GPT-4 / Claude for conversational recommendations
- [ ] **More languages** — French, German query support
- [ ] **Mobile app** — React Native frontend
- [ ] **Real-time data** — scrape live prices from e-commerce sites

---

<div align="center">
Built with MohamedBahaa using Flask · TensorFlow · scikit-learn
</div>
