# Email/SMS Spam Detection System

**Predict whether an incoming text message (email or SMS) is spam or legitimate (ham). Built with Streamlit and a Naïve Bayes model achieving \~97 % accuracy and perfect precision on the validation set.**

## Motivation

This project was built as a hands-on way to apply my machine learning skills to a real-world NLP problem. I wanted to understand the full pipeline — from data cleaning and feature engineering to model selection and deployment. Spam detection is a classic use case that allowed me to practice both backend ML and frontend deployment using Streamlit.

## Features

* Web app – paste or type a message and get an instant spam/ham prediction.
* Built on a TF‑IDF vectorizer + Multinomial Naïve Bayes classifier.
* Exploratory Data Analysis (EDA) dashboards: word clouds, class distribution, common n‑grams.
* Model pipeline saved via `joblib` for reproducible deployment.

## How it Works

1. The raw Kaggle dataset is cleaned (lower‑casing, punctuation removal, stop‑word filtering, tokenisation).
2. Text is transformed into TF‑IDF vectors.
3. A Multinomial Naïve Bayes classifier (tuned via grid‑search) performs the prediction.
4. Streamlit wraps the pipeline in a lightweight UI.

## Tech Stack

| Layer   | Libraries / Tools                                         |
| ------- | --------------------------------------------------------- |
| UI      | **Streamlit**                                             |
| Data    | `pandas`, `numpy`, `nltk`                                 |
| ML      | `scikit‑learn` (Naïve Bayes, `TfidfVectorizer`), `joblib` |
| Visuals | `matplotlib`, `seaborn`, `wordcloud`                      |

## Quick Start (local)

```bash
# clone the repo
git clone https://github.com/karann18/email-sms-spam-detection.git
cd email-sms-spam-detection

# (optional) create & activate virtualenv
python -m venv .venv
.\.venv\Scripts\activate           # Windows
# source .venv/bin/activate        # macOS/Linux

# install dependencies
pip install -r requirements.txt

# download necessary NLTK data (only once)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# run the app
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

> ⚠️ Make sure `model.pkl` and `vectorizer.pkl` are in the same folder as `app.py`.
> 📁 `dataset.csv` is not required to run the app — it's only used during training.

## File Structure

```
├── app.py                  # Streamlit front‑end
├── model.pkl               # trained Naïve Bayes model
├── vectorizer.pkl          # TF‑IDF vectorizer
├── requirements.txt
├── dataset.csv             # raw data (optional)
└── README.md
```

## Dataset

* **SMS Spam Collection** – available on Kaggle: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Live Demo

The project is deployed on Streamlit Community Cloud.
➡️ **URL:** [https://email-sms-spam-detection-karann18.streamlit.app](https://email-sms-spam-detection-karann18.streamlit.app)

## Author

**Karan Vishwakarma** – [GitHub](https://github.com/karann18)

## License

This project is licensed under the MIT License – see `LICENSE` for details.
