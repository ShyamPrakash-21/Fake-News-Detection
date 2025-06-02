# Fake News Detection with Python 📰🚫

This project uses **TF-IDF Vectorization** and **Passive Aggressive Classifier** to detect fake news.

## Dataset
The dataset `news.csv` contains labeled news articles with the following columns:
- `id`
- `title`
- `text`
- `label` (REAL or FAKE)

## Requirements
Install the following Python packages:
```bash
pip install -r requirements.txt
```

## Run the Code
You can run the Jupyter Notebook:
```bash
jupyter lab
```

Or the script:
```bash
python fake_news_detection.py
```

## Accuracy
Achieved ~92.82% accuracy using TfidfVectorizer and PassiveAggressiveClassifier.

## Confusion Matrix
Provides insights into True Positives, True Negatives, False Positives, and False Negatives.
