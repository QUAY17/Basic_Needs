import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from transformers import pipeline
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
    if text in ["no", "nan", "na", "n/a", "doesn't apply", "not applicable", "im not too sure", "not at this time", "no thank you", "not sure", "not really sure", "other"]:
        return "N/A", 0.0
    
    # Custom rules based on question context
    if question == "How is food or housing insecurity affecting your work?":
        if any(phrase in text for phrase in ["not", "do not", "does not", "no", "not affecting", "not affected", "it is not", "not applicable"]):
            return "POSITIVE", score
        elif any(word in text for word in ["struggle", "struggling", "focus", "distracted", "low income", "concentrate", "cannot afford", "suffers", "unable", "hard", "difficult", "worry", "concern", "issue", "hungry", "tired", "stress", "prices", "anxiety", "lack", "cost of rent", "can't afford", "depressed", "inflation", "expensive", "yes", "headache", "sick", "multiple jobs", "affect the quality of my work", "feeling sleepy and hungry at work", "can't concentrate", "distracted", "housing in an apartment", "worries me", "not enjoy my job", "rise of costs", "unreliable", "no food fresh", "no choice", "i'm not paid enough"]):
            return "NEGATIVE", score
    
    elif question == "What could your college or university do to address food and housing insecurity?":
        if any(phrase in text for phrase in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand", "opening a food pantry", "opening a small pantry", "good assistance", "very helpful", "made active strides to provide"]):
            return "POSITIVE", score
        elif any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits", "raise pay", "make it affordable", "need", "pay fair wages", "better wages", "lower food prices", "nightmares", "increase salary"]):
            return "NEGATIVE", score
    
    elif question == "Is there anything else you would like to share?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "useless", "pointless", "wasting our time", "waste of time", "cost of living", "grossly outpaced salaries", "inequities", "low income", "disgusting", "stress", "struggling", "no principles", "high costs of living", "low salary", "hostility", "safety", "loss of health insurance", "low enrollment" ]):
            return "NEGATIVE", score
        elif any(word in text for word in ["positive", "good", "well", "happy", "satisfied", "help", "support", "assist", "care", "benefit", "appreciate", "glad", "look forward", "amazed", "impressed", "appreciate", "thank you"]):
            return "POSITIVE", score

    # Custom rules for new questions
    elif question == "Please select the reasons for not visiting the campus food pantry.":
        if any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits" ,"inconvenient", "eligible", "dietary needs"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand", "do not need assistance", "i do not need assistance with obtaining food and household supplies", "i visit another food pantry/food bank in my community", "i don't want other people to see me and know that i am food insecure", "other students need this help more than i do"]):
            return "POSITIVE", score

    elif question == "What are your thoughts about food availability on your campus?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "insufficient", "unhealthy", "limited", "expensive", "wish", "ought", "more", "less expensive", "cheaper", "low cost", "subsidized", "bad", "need healthy", "unhealthy", "can't pay", "can't afford", "overly priced", "terrible", "not good", "options are needed", "low quality"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["good", "great", "getting better", "improve", "improving", "affordable", "nice", "positive", "appreciate", "not bad",  "available", "delicious", "enjoy", "working on programs", "prepared well"]):
            return "POSITIVE", score

    elif question == "Please share why you feel unsafe?":
        if any(word in text for word in ["concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", "unsafe", "dangerous", "fear", "scared", "shootings", "crime", "violence", "abusive", "verbal abuse", "domestic violence", "vagrants", "gunfire", "shootings", "stalker", "break ins", "drugs", "homeless", "drug dealer", "property crime", "threat", "shady", "unsafe neighborhood", "threat", "steal", "broke in", "scary", "unhoused", "stalk", "lacks privacy", "domestic relations", "challenging", "my dad", "my ex", "my husband", "my family"]):
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

    # Save results to CSV
    sentiment_results.to_csv(f"sentiment_analysis_results_{column}.csv", index=False)

# Function to extract keywords
def extract_keywords(df, column, model_name='all-mpnet-base-v2', ngram_range=(1, 2), top_n=5):
    progress_logger.info(f"Extracting keywords for column: {column}")
    embedding_model = SentenceTransformer(model_name)
    keyword_extractor = KeyBERT(model=embedding_model)
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()  # Filter out non-informative responses
    
    if not texts:
        progress_logger.warning(f"No valid responses for column: {column}. Skipping keyword extraction and word cloud generation.")
        return
    
    keyword_results = [keyword_extractor.extract_keywords(text, keyphrase_ngram_range=ngram_range, stop_words='english', top_n=top_n) for text in texts]

    # Save results to CSV
    keyword_summary = count_keywords(keyword_results)
    keyword_summary.to_csv(f"keyword_analysis_results_{column}.csv", index=False)

    # Generate and save wordcloud
    keywords_text = " ".join([keyword for keywords in keyword_results for keyword, _ in keywords])
    if keywords_text.strip():
        wordcloud_img = generate_wordcloud(keywords_text)
    else:
        progress_logger.warning(f"No keywords found for column: {column}. Skipping word cloud generation.")

    progress_logger.info(f"Keyword extraction completed for column: {column}")

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
    if not text.strip():
        progress_logger.warning("Empty text provided. Cannot generate word cloud.")
        return None
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Function to analyze topics
def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=5, ngram_range=(1, 2)):
    progress_logger.info(f"Analyzing topics for column: {column}")
    embedding_model = SentenceTransformer(model_name)
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()  # Filter out non-informative responses
    
    if len(texts) == 0:  # Check if there are no responses
        progress_logger.warning(f"No valid responses for column: {column}")
        return
    
    vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size, vectorizer_model=vectorizer_model)
    topics, _ = topic_model.fit_transform(texts)
    
    topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
    
    # Save topic info
    topic_info.to_csv(f"topic_analysis_results_{column}.csv", index=False)
    
    # Save topic summary
    with open(f"topic_summary_{column}.txt", "w") as f:
        for topic in topic_summary:
            f.write(f"Topic {topic['Topic']} (Count: {topic['Count']})\n")
            f.write("Examples:\n")
            for example in topic['Examples']:
                f.write(f"- {example}\n")
            f.write("\n")
    
    progress_logger.info(f"Topic analysis completed for column: {column}")

# Function to summarize topics
def summarize_topics(topic_model, topics, texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    for topic in topic_info['Topic']:
        if topic == -1:
            continue
        topic_count = sum(1 for t in topics if t == topic)
        topic_texts = [text for t, text in zip(topics, texts) if t == topic]
        examples = topic_texts[:5]
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'Examples': examples
        })
    
    return topic_info, topic_summary

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

    for question, col in question_mapping.items():
        df[col] = df[col].apply(preprocess_text)
        analyze_sentiment(df, col, question)
        extract_keywords(df, col)
        
        # Only analyze topics if there are valid responses
        if not df[col].dropna().empty:
            analyze_topics(df, col)
        else:
            progress_logger.warning(f"No valid responses for column: {col}. Skipping topic analysis.")