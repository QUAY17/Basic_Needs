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
import os
import logging

# Set environment variable to disable parallelism warning from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Create handlers
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.DEBUG)  # Log debug and higher to a file

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)  # Only log info and higher to stderr

# Create formatters and add them to the handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Suppress debug logs from external libraries
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('dash').setLevel(logging.WARNING)

# Create a logger for progress updates
progress_logger = logging.getLogger("progress")
progress_logger.setLevel(logging.INFO)

# Set random seed for reproducibility
np.random.seed(42)

def create_dash_app(df, question_mapping):
    # Your Dash app creation code
    app = Dash(__name__)
    # Setup app layout and callbacks
    return app

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower().strip()
    return text

# Function to filter out non-informative responses
def filter_responses(text):
    non_informative_responses = [
        "unsure", "not sure", "nothing", "no", "none", "n/a", "N/A", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
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

    # Custom rules for new questions
    elif question == "Please select the reasons for not visiting the campus food pantry.":
        if any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand"]):
            return "POSITIVE", score

    elif question == "What are your thoughts about food availability on your campus?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "insufficient", "unhealthy", "limited", "expensive", "wish", "ought", "more", "less expensive", "cheaper", "low cost", "subsidized", "bad"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["good", "great", "getting better", "improve", "improving", "affordable", "nice", "positive"]):
            return "POSITIVE", score

    elif question == "Please share why you feel unsafe?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "unsafe", "dangerous", "fear", "scared", "shootings", "crime", "violence", "abusive", "verbal abuse", "domestic violence", "vagrants", "gunfire", "shootings", "stalker", "break ins", "drugs", "homeless", "drug dealer", "property crime", "threat", "shady", "unsafe neighborhood"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["positive", "good", "well", "happy", "satisfied", "safe", "secure", "protected"]):
            return "POSITIVE", score

    elif question == "Please explain why it is difficult to find housing either on-campus or off-campus?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["positive", "good", "well", "happy", "satisfied", "help", "support", "assist", "care", "benefit"]):
            return "POSITIVE", score
    
    # Default to model's sentiment if no rule applies
    if score < threshold:
        return "N/A", score
    return sentiment, score

# Function to analyze sentiment
def analyze_sentiment(df, column, question):
    progress_logger.info(f"Analyzing sentiment for question: {question}...")
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
    progress_logger.info(f"Extracting keywords...")
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
    progress_logger.info(f"Generating wordcloud...")
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Function to analyze topics
def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=5, ngram_range=(1, 2)):
    progress_logger.info(f"Analyzing topics...")
    embedding_model = SentenceTransformer(model_name)
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()  # Filter out non-informative responses
    if len(texts) == 0:  # Check if there are no responses
        return None, None, texts
    vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words='english')
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
        # Check if the column has any responses
        if df[col].dropna().apply(filter_responses).empty:
            tab_content = html.Div([
                html.H2(question),
                html.P("No responses were provided for this question.")
            ])
        else:
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
            if topic_model is None:  # Handle case where there are no responses
                tab_content = html.Div([
                    html.H2(question),
                    html.P("No responses were provided for this question.")
                ])
            else:
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
    csv_file = "statewide_facultystaff_24.csv"
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    progress_logger.info("Loading data...")
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3",
        "Please select the reasons for not visiting the campus food pantry.": "Foodpantry_reasons",
        "What are your thoughts about food availability on your campus?": "Foodavail",
        "Please share why you feel unsafe?": "Unsafe_why",
        "Please explain why it is difficult to find housing either on-campus or off-campus?": "Housingdiff_why"
    }
    for col in question_mapping.values():
        df[col] = df[col].apply(preprocess_text)
    
    progress_logger.info("Creating Dash app...")
    app = create_dash_app(df, question_mapping)
    progress_logger.info("Starting Dash server...")
    app.run_server(debug=True, host='0.0.0.0', port=8080, use_reloader=False)
