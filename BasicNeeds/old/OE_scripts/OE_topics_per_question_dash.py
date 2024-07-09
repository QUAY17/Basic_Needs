import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from dash import Dash, html, dcc
import plotly.express as px
import plotly.graph_objects as go
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

# Function to perform topic modeling on a single open-ended question
def analyze_topics(df, column, model_name='all-mpnet-base-v2'):
    # Load the Sentence Transformer model
    embedding_model = SentenceTransformer(model_name)
    
    # Get the responses for the specific open-ended question
    texts = df[column].dropna().apply(preprocess_text).tolist()
    texts = [text for text in texts if text and text not in ["nan", "no", "na", "n/a", "doesn't apply", "not applicable"]]
    
    # Perform topic modeling using BERTopic
    topic_model = BERTopic(embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(texts)
    
    return topic_model, topics, texts

# Function to summarize topics
def summarize_topics(topic_model, topics, texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    for topic in topic_info['Topic']:
        if topic == -1:  # Skip the outlier topic
            continue
        topic_count = sum([1 for t in topics if t == topic])
        topic_texts = [texts[i] for i, t in enumerate(topics) if t == topic]
        topic_ids = [i for i, t in enumerate(topics) if t == topic]
        examples = topic_texts[:5]  # Get up to 5 example texts
        
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'IDs': topic_ids,
            'Examples': examples
        })
    
    return topic_info, topic_summary

# Function to create Dash app for interactive visualizations
def create_dash_app(topic_info, topic_summary, question):
    app = Dash(__name__)

    fig = go.Figure()

    for topic in topic_summary:
        examples_text = "<br>".join(topic['Examples'])
        fig.add_trace(go.Bar(
            x=[f"Topic {topic['Topic']}"],
            y=[topic['Count']],
            name=f"Topic {topic['Topic']}",
            text=[f"Count: {topic['Count']}<br>Examples:<br>{examples_text}"],
            hoverinfo='text'
        ))

    fig.update_layout(
        title=f"Topic Analysis Results for Question: {question}",
        xaxis_title="Topics",
        yaxis_title="Count of Responses",
        hovermode="closest"
    )

    app.layout = html.Div([
        html.H1(f"Topic Analysis for Question: {question}"),
        dcc.Graph(figure=fig)
    ])

    return app

# Example usage
if __name__ == "__main__":
    # Example to demonstrate usage; replace with actual file path and columns
    csv_file = "data/statewide_facultystaff_24.csv"
    
    # Load and preprocess the dataset
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3"
    }
    
    for question, col in question_mapping.items():
        df[col] = df[col].apply(preprocess_text)
        
        # Perform topic analysis for the current question
        topic_model, topics, texts = analyze_topics(df, col)
        
        # Summarize topics
        topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
        
        # Create Dash app for the current question
        app = create_dash_app(topic_info, topic_summary, question)
        
        app.run_server(debug=True)
        
        print(f"Topic analysis results for {question} have been processed.")
