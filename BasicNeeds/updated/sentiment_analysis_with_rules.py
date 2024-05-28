import pandas as pd
import numpy as np
from transformers import pipeline
from dash import Dash, dcc, html
import plotly.express as px

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower().strip()
    return text

# Function to classify sentiment and handle edge cases
def classify_sentiment(text, sentiment, score, question, threshold=0.6):
    no_sentiment_phrases = ["no", "nan", "na", "n/a", "doesn't apply", "not applicable", "im not too sure", "not at this time", "no thank you"]
    if text in no_sentiment_phrases:
        return "N/A", 0.0
    
    # Custom rules based on question context
    if question == "How is food or housing insecurity affecting your work?":
        if any(phrase in text for phrase in ["not", "do not", "does not", "no"]):
            return "POSITIVE", score
        elif any(word in text for word in ["affect", "impact", "struggle", "unable", "hard", "difficult", "worry", "concern", "issue"]):
            return "NEGATIVE", score
    
    elif question == "What could your college or university do to address food and housing insecurity?":
        if any(phrase in text for phrase in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand", "opening a food pantry", "opening a small pantry"]):
            return "POSITIVE", score
        elif any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits"]):
            return "NEGATIVE", score
    
    elif question == "Is there anything else you would like to share?":
        if text in no_sentiment_phrases:
            return "N/A", 0.0
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "useless", "pointless", "wasting our time", "waste of time", "cost of living", "grossly outpaced salaries"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["positive", "good", "well", "happy", "satisfied", "help", "support", "assist", "care", "benefit", "appreciate", "glad", "look forward"]):
            return "POSITIVE", score
    
    # Default to model's sentiment if no rule applies
    if score < threshold:
        return "N/A", score
    return sentiment, score

# Function to analyze sentiment
def analyze_sentiment(df, column, question):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = []
    for idx, text in df[column].dropna().items():
        result = sentiment_pipeline(text)[0]
        sentiment, score = classify_sentiment(text, result['label'], result['score'], question)
        sentiments.append({'ID': df['ID'][idx], 'text': text, 'label': sentiment, 'score': score})
    sentiment_results = pd.DataFrame(sentiments)
    return sentiment_results

# Example usage
if __name__ == "__main__":
    csv_file = "data/statewide_facultystaff_24.csv"
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3"
    }
    for col in question_mapping.values():
        df[col] = df[col].apply(preprocess_text)
    
    for question, col in question_mapping.items():
        sentiment_results = analyze_sentiment(df, col, question)
        sentiment_results.to_csv(f"sentiment_analysis_results_{col}.csv", index=False)
        sentiment_fig = px.pie(sentiment_results, names='label', title=f'Sentiment Distribution for {question}')
        sentiment_fig.show()
