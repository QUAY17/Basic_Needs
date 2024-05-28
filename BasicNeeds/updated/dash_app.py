import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from transformers import pipeline
from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower().strip()
    return text

# Function to filter out non-informative responses
def filter_responses(text):
    non_informative_responses = [
        "unsure", "not sure", "nothing", "no", "none", "n/a", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "iâ€™m not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not applicable", "not at this time", "no thank you", "na"
    ]
    if any(response in text for response in non_informative_responses):
        return False
    return True

# Function to classify sentiment and handle edge cases
def classify_sentiment(text, sentiment, score, question, threshold=0.6):
    if text in ["no", "nan", "na", "n/a", "doesn't apply", "not applicable", "im not too sure", "not at this time", "no thank you"]:
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
        if text in ["no", "nan", "na", "n/a", "doesn't apply", "not applicable", "im not too sure", "not at this time", "no thank you"]:
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

# Function to extract keywords
def extract_keywords(df, column, model_name='all-mpnet-base-v2', ngram_range=(1, 2), top_n=5):
    embedding_model = SentenceTransformer(model_name)
    keyword_extractor = KeyBERT(model=embedding_model)
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()  # Filter out non-informative responses
    keyword_results = [keyword_extractor.extract_keywords(text, keyphrase_ngram_range=ngram_range, stop_words='english', top_n=top_n) for text in texts]
    return keyword_results

# Function to count keywords
def count_keywords(keyword_results):
    keyword_counter = Counter()
    for keywords in keyword_results:
        keyword_counter.update([keyword for keyword, _ in keywords])
    keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
    return keyword_summary

# Function to generate word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Function to analyze topics
def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=5, ngram_range=(1, 2)):
    embedding_model = SentenceTransformer(model_name)
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()  # Filter out non-informative responses
    vectorizer_model = CountVectorizer(ngram_range=ngram_range)
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size, vectorizer_model=vectorizer_model)
    topics, probs = topic_model.fit_transform(texts)
    return topic_model, topics, texts

# Function to summarize topics
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

# Create Dash app
def create_dash_app(df, question_mapping):
    app = Dash(__name__)
    
    tabs = []
    
    for question, col in question_mapping.items():
        # Sentiment Analysis
        sentiment_results = analyze_sentiment(df, col, question)
        sentiment_results.to_csv(f"sentiment_analysis_results_{col}.csv", index=False)
        sentiment_fig = px.pie(sentiment_results, names='label', title=f'Sentiment Distribution for {question}')
        
        # Keyword Extraction
        keyword_results = extract_keywords(df, col)
        keyword_summary = count_keywords(keyword_results)
        keyword_summary.to_csv(f"keyword_analysis_results_{col}.csv", index=False)
        keyword_fig = px.bar(keyword_summary.head(20), x='Keyword', y='Count', title=f'Top 20 Keywords for {question}')
        
        keywords_text = " ".join([keyword for keywords in keyword_results for keyword, _ in keywords])
        wordcloud_img = generate_wordcloud(keywords_text)
        
        # Topic Analysis
        topic_model, topics, texts = analyze_topics(df, col, min_topic_size=5)  # Adjusted min_topic_size for better separation
        topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
        topic_info.to_csv(f"topic_analysis_results_{col}.csv", index=False)
        topic_bar_fig = go.Figure()
        for topic in topic_summary:
            examples_text = "<br>".join(topic['Examples'])
            topic_bar_fig.add_trace(go.Bar(
                x=[f"Topic {topic['Topic']}"],
                y=[topic['Count']],
                name=f"Topic {topic['Topic']}",
                text=[f"Count: {topic['Count']}<br>Examples:<br>{examples_text}"],
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
    app.run_server(debug=True, port=8052)
