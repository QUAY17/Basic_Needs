import pandas as pd
import re
import numpy as np
import random
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT
import string

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Function to remove BOM from a CSV file
def remove_bom(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(file_path, 'wb') as file:
        file.write(content)

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Load and preprocess the dataset
csv_file = "data/statewide_facultystaff_24.csv"
remove_bom(csv_file)
df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

# Apply preprocessing
oe_columns = ["OE1", "OE2", "OE3"]
for col in oe_columns:
    df[col] = df[col].apply(preprocess_text)

# Sentiment Analysis
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
sentiment_results = {col: [] for col in oe_columns}

for _, row in df.iterrows():
    for col in oe_columns:
        if pd.notna(row[col]):
            sentiment = sentiment_pipeline(row[col])[0]
            sentiment_results[col].append((row["ID"], row[col], sentiment["label"], sentiment["score"]))
        else:
            sentiment_results[col].append((row["ID"], row[col], "N/A", 0.0))

# Keyword Extraction using KeyBERT
model_name = 'all-mpnet-base-v2'  # More contemporary model from Sentence Transformers
embedding_model = SentenceTransformer(model_name)
keyword_extractor = KeyBERT(model=embedding_model)
keyword_results = {col: [] for col in oe_columns}

for _, row in df.iterrows():
    for col in oe_columns:
        if pd.notna(row[col]):
            keywords = keyword_extractor.extract_keywords(row[col], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
            keyword_results[col].append((row["ID"], row[col], keywords))
        else:
            keyword_results[col].append((row["ID"], row[col], []))

# Topic Modeling using BERTopic
topic_model = BERTopic(embedding_model=embedding_model)

all_texts = df[oe_columns].fillna('').values.flatten()
all_texts = [text for text in all_texts if text]

topics, probs = topic_model.fit_transform(all_texts)
topic_info = topic_model.get_topic_info()

# Save the results
with open("analysis_results.txt", "w") as file:
    file.write("Sentiment Analysis Results:\n")
    for col, results in sentiment_results.items():
        file.write(f"Sentiment Analysis for {col}:\n")
        for id, text, sentiment, score in results:
            file.write(f"ID: {id}\nText: {text}\nSentiment: {sentiment}\nScore: {score}\n\n")

    file.write("Keyword Extraction Results:\n")
    for col, results in keyword_results.items():
        file.write(f"Keyword Extraction for {col}:\n")
        for id, text, keywords in results:
            file.write(f"ID: {id}\nText: {text}\nKeywords: {keywords}\n\n")

    file.write("Topic Modeling Results:\n")
    file.write(topic_info.to_string())

print("Analysis results have been written to 'analysis_results.txt'.")
