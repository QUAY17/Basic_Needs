import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
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

# Function to perform keyword extraction on open-ended questions
def extract_keywords(df, oe_columns, model_name='all-mpnet-base-v2'):
    # Load the Sentence Transformer model
    embedding_model = SentenceTransformer(model_name)
    keyword_extractor = KeyBERT(model=embedding_model)
    keyword_results = {col: [] for col in oe_columns}

    for _, row in df.iterrows():
        for col in oe_columns:
            if pd.notna(row[col]) and row[col] not in ["nan", "no", "na", "n/a", "doesn't apply", "not applicable"]:
                keywords = keyword_extractor.extract_keywords(row[col], keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
                keyword_results[col].append((row["ID"], row[col], keywords))
            else:
                keyword_results[col].append((row["ID"], row[col], []))
    
    return keyword_results

# Function to save keyword extraction results
def save_keyword_results(keyword_results, filename="keyword_extraction_results.txt"):
    with open(filename, "w") as file:
        file.write("Keyword Extraction Results:\n")
        for col, results in keyword_results.items():
            file.write(f"Keyword Extraction for {col}:\n")
            for id, text, keywords in results:
                if text not in ["nan", "no", "na", "n/a", "doesn't apply", "not applicable"]:
                    file.write(f"ID: {id}\nText: {text}\nKeywords: {keywords}\n\n")

# Function to count keywords and rank their usage
def count_keywords(keyword_results):
    keyword_counter = Counter()

    for col, results in keyword_results.items():
        for _, _, keywords in results:
            keyword_counter.update([keyword for keyword, _ in keywords])

    keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
    return keyword_summary

# Function to plot keyword frequency
def plot_keyword_frequency(keyword_results, question_mapping, top_n=20):
    for question, col in question_mapping.items():
        keyword_counter = Counter()

        for _, _, keywords in keyword_results[col]:
            keyword_counter.update([keyword for keyword, _ in keywords])
        
        keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False).head(top_n)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=keyword_summary['Keyword'],
            y=keyword_summary['Count'],
            text=keyword_summary['Count'],
            textposition='auto'
        ))
        fig.update_layout(
            title=f"Top {top_n} Keywords for Question: {question}",
            xaxis_title="Keywords",
            yaxis_title="Count",
            hovermode="closest"
        )
        fig.show()

# Example usage
if __name__ == "__main__":
    # Example to demonstrate usage; replace with actual file path and columns
    csv_file = "data/statewide_facultystaff_24.csv"
    
    # Load and preprocess the dataset
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    oe_columns = ["OE1", "OE2", "OE3"]
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3"
    }
    for col in oe_columns:
        df[col] = df[col].apply(preprocess_text)
    
    # Perform keyword extraction
    keyword_results = extract_keywords(df, oe_columns)
    
    # Save the keyword extraction results
    save_keyword_results(keyword_results)
    
    # Plot keyword frequency for each question
    plot_keyword_frequency(keyword_results, question_mapping)
    
    print("Keyword extraction results have been written to 'keyword_extraction_results.txt'.")
    print("Keyword frequency visualizations have been generated.")
