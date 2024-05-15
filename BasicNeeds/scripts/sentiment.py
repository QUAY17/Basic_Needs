import warnings
import pandas as pd
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import pipeline

# Suppress specific warnings or all warnings
warnings.filterwarnings("ignore")

DB_FAISS_PATH = "vectorstore/db_faiss"

# Function to remove BOM from a CSV file
def remove_bom(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(file_path, 'wb') as file:
        file.write(content)

# Remove BOM from the CSV file if present
csv_file = "data/statewide_facultystaff_24.csv"
remove_bom(csv_file)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

# Initialize the sentiment analysis pipeline with a specified model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment for OE questions
def analyze_sentiment(df):
    oe_columns = ["OE1", "OE2", "OE3"]
    sentiment_results = {col: [] for col in oe_columns}

    for _, row in df.iterrows():
        for col in oe_columns:
            if pd.notna(row[col]):
                sentiment = sentiment_pipeline(row[col])[0]
                sentiment_results[col].append((row["ID"], row[col], sentiment["label"], sentiment["score"]))
            else:
                sentiment_results[col].append((row["ID"], row[col], "N/A", 0.0))

    return sentiment_results

# Analyze sentiment for OE questions
sentiment_results = analyze_sentiment(df)

# Save the sentiment results to a file
with open("sentiment_analysis_results.txt", "w") as file:
    for col, results in sentiment_results.items():
        file.write(f"Sentiment Analysis for {col}:\n")
        for id, text, sentiment, score in results:
            file.write(f"ID: {id}\nText: {text}\nSentiment: {sentiment}\nScore: {score}\n\n")

print("Sentiment analysis results have been written to 'sentiment_analysis_results.txt'.")
