import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from transformers import pipeline
from dash import Dash, dcc, html
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
from collections import Counter
import base64
from io import BytesIO
import string

# Set random seed for reproducibility
np.random.seed(42)

# Preprocessing and analysis functions
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def analyze_sentiment(df, column):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = df[column].dropna().apply(lambda x: sentiment_pipeline(x)[0] if x else {"label": "N/A", "score": 0.0})
    sentiment_results = pd.DataFrame(sentiments.tolist())
    sentiment_results['text'] = df[column]
    return sentiment_results

def extract_keywords(df, column, model_name='all-mpnet-base-v2'):
    embedding_model = SentenceTransformer(model_name)
    keyword_extractor = KeyBERT(model=embedding_model)
    texts = df[column].dropna().apply(preprocess_text).tolist()
    keyword_results = [keyword_extractor.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5) for text in texts]
    return keyword_results

def analyze_topics(df, column, model_name='all-mpnet-base-v2'):
    embedding_model = SentenceTransformer(model_name)
    texts = df[column].dropna().apply(preprocess_text).tolist()
    topic_model = BERTopic(embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, texts

def summarize_topics(topic_model, topics, texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    for topic in topic_info['Topic']:
        if topic == -1:
            continue
        topic_count = sum([1 for t in topics if t == topic])
        topic_texts = [texts[i] for i, t in enumerate(topics) if t == topic]
        examples = topic_texts[:5]
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'Examples': examples
        })
    return topic_info, topic_summary

def count_keywords(keyword_results):
    keyword_counter = Counter()
    for keywords in keyword_results:
        keyword_counter.update([keyword for keyword, _ in keywords])
    keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
    return keyword_summary

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Function to create Dash app
def create_dash_app(df, question_mapping):
    app = Dash(__name__)
    
    tabs = []
    
    for question, col in question_mapping.items():
        # Sentiment Analysis
        sentiment_results = analyze_sentiment(df, col)
        sentiment_fig = px.pie(sentiment_results, names='label', title=f'Sentiment Distribution for {question}')
        
        sentiment_scatter = px.scatter(sentiment_results, x='text', y='score', color='label', title=f'Sentiment Scores for {question}')
        
        # Keyword Extraction
        keyword_results = extract_keywords(df, col)
        keyword_summary = count_keywords(keyword_results)
        keyword_fig = px.bar(keyword_summary.head(20), x='Keyword', y='Count', title=f'Top 20 Keywords for {question}')
        
        keywords_text = " ".join([keyword for keywords in keyword_results for keyword, _ in keywords])
        wordcloud_img = generate_wordcloud(keywords_text)
        
        # Topic Analysis
        topic_model, topics, texts = analyze_topics(df, col)
        topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
        
        topic_bar_fig = go.Figure()
        for topic in topic_summary:
            topic_bar_fig.add_trace(go.Bar(
                x=[f"Topic {topic['Topic']}"],
                y=[topic['Count']],
                name=f"Topic {topic['Topic']}",
                text=[f"Count: {topic['Count']}<br>Examples:<br>" + "<br>".join(topic['Examples'])],
                hoverinfo='text'
            ))
        topic_bar_fig.update_layout(
            title=f"Topic Analysis Results for {question}",
            xaxis_title="Topics",
            yaxis_title="Count of Responses",
            hovermode="closest"
        )
        
        topic_example_table = []
        for topic in topic_summary:
            topic_example_table.append(html.H4(f"Topic {topic['Topic']} (Count: {topic['Count']})"))
            for example in topic['Examples']:
                topic_example_table.append(html.P(example))
        
        tab_content = html.Div([
            html.H2(question),
            dcc.Graph(figure=sentiment_fig),
            dcc.Graph(figure=sentiment_scatter),
            dcc.Graph(figure=keyword_fig),
            html.Img(src=f"data:image/png;base64,{wordcloud_img}", style={'width': '100%', 'height': 'auto'}),
            dcc.Graph(figure=topic_bar_fig),
            html.Div(topic_example_table)
        ])
        
        tabs.append(dcc.Tab(label=question, children=tab_content))
    
    app.layout = html.Div([
        html.H1("Comprehensive Analysis of Open-Ended Questions"),
        dcc.Tabs(tabs)
    ])
    
    return app

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
    
    app = create_dash_app(df, question_mapping)
    app.run_server(debug=True, port=8052)  # Change the port to avoid conflicts
