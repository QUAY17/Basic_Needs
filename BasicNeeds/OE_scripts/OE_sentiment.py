import pandas as pd
import numpy as np
from transformers import pipeline
import string

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to classify sentiment and handle edge cases
def classify_sentiment(text, sentiment, score):
    if text in ["no", "it is not", "na", "nan", "n/a", "doesn't apply", "not applicable"]:
        return "N/A", 0.0
    return sentiment, score

# Function to perform sentiment analysis on open-ended questions
def analyze_sentiment(df, oe_columns):
    # Initialize Sentiment Analysis Pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiment_results = {col: [] for col in oe_columns}

    for _, row in df.iterrows():
        for col in oe_columns:
            if pd.notna(row[col]):
                result = sentiment_pipeline(row[col])[0]
                sentiment, score = classify_sentiment(row[col], result["label"], result["score"])
                sentiment_results[col].append((row["ID"], row[col], sentiment, score))
            else:
                sentiment_results[col].append((row["ID"], row[col], "N/A", 0.0))

    return sentiment_results

# Function to collect summary statistics
def summarize_sentiment(sentiment_results):
    summary_data = []

    for col, results in sentiment_results.items():
        sentiments = [res[2] for res in results]
        scores = [res[3] for res in results]
        summary_data.append({
            'question': col,
            'total_responses': len(sentiments),
            'positive': sentiments.count('POSITIVE'),
            'negative': sentiments.count('NEGATIVE'),
            'n/a': sentiments.count('N/A'),
            'average_score': np.mean(scores)
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

# Function to plot sentiment summary
def plot_summary(summary_df):
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot for sentiment distribution
    summary_df[['positive', 'negative', 'n/a']].plot(kind='bar', ax=axes[0])
    axes[0].set_title('Sentiment Distribution')
    axes[0].set_xlabel('Open-Ended Questions')
    axes[0].set_ylabel('Count')
    axes[0].legend(title='Sentiment')
    
    # Pie chart for average scores
    summary_df.set_index('question')['average_score'].plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
    axes[1].set_ylabel('')
    axes[1].set_title('Average Confidence Scores')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example to demonstrate usage; replace with actual file path and columns
    csv_file = "data/statewide_facultystaff_24.csv"
    
    # Load and preprocess the dataset
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    oe_columns = ["OE1", "OE2", "OE3"]
    for col in oe_columns:
        df[col] = df[col].apply(preprocess_text)
    
    # Perform sentiment analysis
    sentiment_results = analyze_sentiment(df, oe_columns)
    
    # Summarize sentiment results
    summary_df = summarize_sentiment(sentiment_results)
    
    # Plot summary
    plot_summary(summary_df)
    
    # Save the sentiment results to a file
    with open("sentiment_analysis_results.txt", "w") as file:
        file.write("Sentiment Analysis Results:\n")
        for col, results in sentiment_results.items():
            file.write(f"Sentiment Analysis for {col}:\n")
            for id, text, sentiment, score in results:
                file.write(f"ID: {id}\nText: {text}\nSentiment: {sentiment}\nScore: {score}\n\n")
    
    print("Sentiment analysis results have been written to 'sentiment_analysis_results.txt'.")
