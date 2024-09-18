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
import sys

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
        "unsure", "not sure", "nothing", "no", "none", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not at this time", "no thank you"
    ]
    return text not in non_informative_responses
    
# Custom rules based on question context
def classify_sentiment(text, sentiment, score, question, threshold=0.6):
    if question == "How is food or housing insecurity affecting your work?":
        # Multi-word phrase matching (positive)
        positive_phrases = [
            "doesn't affect my work", "not affecting me", "no impact on my work",
            "no effect", "no worry"
        ]
        
        # Multi-word phrase matching (negative)
        negative_phrases = [
            "struggling to focus", "cannot afford", "unable to concentrate", 
            "affect the quality of my work", "feeling sleepy and hungry at work", 
            "worries me about housing", "concerned about rent", "hungry at work"
        ]
        
        # Check for multi-word phrases first
        if any(phrase in text for phrase in positive_phrases):
            return "POSITIVE", score
        elif any(phrase in text for phrase in negative_phrases):
            return "NEGATIVE", score
        
        # Single-word matching (positive)
        if any(word in text for word in ["doesn't", "doesnt", "not", "do not", "does not", "no", "not affecting", "not affected", "it is not", "not applicable", "na", "N/A", "n/a"]):
            return "POSITIVE", score
        
        # Single-word matching (negative)
        if any(word in text for word in ["struggle", "struggling", "focus", "distracted", "low income", "concentrate", "cannot afford", "suffers", "unable", "hard", "difficult", "worry", "concern", "issue", "hungry", "tired", "stress", "prices", "anxiety", "lack", "cost of rent", "can't afford", "depressed", "inflation", "expensive", "yes", "sometimes", "headache", "sick", "multiple jobs", "distracted", "housing in an apartment", "worries me", "not enjoy my job", "rise of costs", "unreliable", "no food fresh", "no choice", "i'm not paid enough"]):
            return "NEGATIVE", score

    elif question == "What could your college or university do to address food and housing insecurity?":
        if any(phrase in text for phrase in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand", "opening a food pantry", "opening a small pantry", "good assistance", "very helpful", "made active strides to provide"]):
            return "POSITIVE", score
        elif any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits", "raise pay", "make it affordable", "need", "pay fair wages", "better wages", "lower food prices", "nightmares", "increase salary"]):
            return "NEGATIVE", score
    
    elif question == "Is there anything else you would like to share?":
        # Multi-word negative phrases
        negative_phrases = [
            "wasting our time", "waste of time", "grossly outpaced salaries", "high costs of living", "loss of health insurance",
            "low enrollment", "struggling with stress", "no principles", "hostility at work", "lack of safety", "no support", 
            "no help", "no resources", "not enough support", "not enough pay"
        ]
        
        # Multi-word positive phrases
        positive_phrases = [
            "thank you for", "very impressed", "appreciate the support", "glad to have", "look forward to", 
            "amazed by the work", "well organized", "help students", "no issues", "not a problem", "no concerns"
        ]
        
        # Single words indicating negative sentiment in context
        negative_words = [
            "concern", "worry", "issue", "problem", "struggle", "challenge", "difficult", "hard", "impact", "affect", 
            "useless", "pointless", "cost of living", "inequities", "low income", "disgusting", "stress", "struggling", 
            "no principles", "low salary", "hostility", "safety", "loss of health insurance", "low enrollment", "pay", "need"
        ]
        
        # Single words indicating positive sentiment in context
        positive_words = [
            "thank", "help", "positive", "good", "well", "happy", "satisfied", "support", "assist", "care", "benefit", 
            "appreciate", "glad", "look forward", "amazed", "impressed", "thank you", "students", "no issues", "not a problem"
        ]
        
        # Contextualize ambiguous words like "no" and "not"
        ambiguous_words = ["no", "not"]
        if any(word in text.lower() for word in ambiguous_words):
            # If "no" or "not" is used in a positive context
            if any(phrase in text.lower() for phrase in ["no issues", "no concerns", "not a problem", "not concerned"]):
                return "POSITIVE", score
            # If "no" or "not" is used in a negative context
            elif any(phrase in text.lower() for phrase in ["no support", "no help", "not enough pay", "not enough support"]):
                return "NEGATIVE", score

        # Check for multi-word phrases first
        if any(phrase in text.lower() for phrase in positive_phrases):
            return "POSITIVE", score
        elif any(phrase in text.lower() for phrase in negative_phrases):
            return "NEGATIVE", score
        
        # Check for single words (positive and negative)
        elif any(word in text.lower() for word in positive_words):
            return "POSITIVE", score
        elif any(word in text.lower() for word in negative_words):
            return "NEGATIVE", score

    # Custom rules for new questions
    elif question == "Please select the reasons for not visiting the campus food pantry.":
        if any(word in text for word in ["don't", "not", "lack", "need", "problem", "compensation", "wages", "pay", "reevaluated", "living wage", "fair wage", "higher wage", "medical benefits" ,"inconvenient", "eligible", "dietary needs"]):
            return "NEGATIVE", score
        elif any(word in text for word in ["help", "support", "provide", "offer", "assist", "resource", "hope", "caring", "ready to give a helping hand", "do not need assistance", "i do not need assistance with obtaining food and household supplies", "i visit another food pantry/food bank in my community", "i don't want other people to see me and know that i am food insecure", "other students need this help more than i do"]):
            return "POSITIVE", score

    elif question == "What are your thoughts about food availability on your campus?":
            # Multi-word negative phrases
            negative_phrases = [
                "not available", "limited options", "too expensive", "no healthy options", "not enough food", 
                "cafeteria is bad", "limited hours", "not many choices", "don't have enough options", "poor food quality"
            ]
            
            # Multi-word positive phrases
            positive_phrases = [
                "good options", "healthy food available", "affordable", "plenty of options", "very good", 
                "food is great", "vending machines are helpful", "good food quality", "healthy meals", "cafeteria is great"
            ]
            
            # Single words indicating negative sentiment
            negative_words = [
                "not", "limited", "expensive", "no", "need", "don't", "hard", "unhealthy", "can't", 
                "difficult", "poor", "few", "bad"
            ]
            
            # Single words indicating positive sentiment
            positive_words = [
                "good", "available", "healthy", "affordable", "options", "meals", "vending", "machines", 
                "great", "improve", "improving", "delicious", "nice", "very"
            ]
            
            # Contextualize ambiguous words like "no" and "not"
            ambiguous_words = ["no", "not"]
            if any(word in text.lower() for word in ambiguous_words):
                # If "no" or "not" is used in a positive context
                if any(phrase in text.lower() for phrase in ["no issues", "not a problem", "no concerns", "not bad"]):
                    return "POSITIVE", score
                # If "no" or "not" is used in a negative context
                elif any(phrase in text.lower() for phrase in ["not available", "no healthy options", "not enough food", "no options"]):
                    return "NEGATIVE", score

            # Check for multi-word phrases first
            if any(phrase in text.lower() for phrase in positive_phrases):
                return "POSITIVE", score
            elif any(phrase in text.lower() for phrase in negative_phrases):
                return "NEGATIVE", score
            
            # Check for single words (positive and negative)
            elif any(word in text.lower() for word in positive_words):
                return "POSITIVE", score
            elif any(word in text.lower() for word in negative_words):
                return "NEGATIVE", score

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
    
    # Default to model's sentiment if no custom rule applies
    if score < threshold:
        return "non-informative", 0.0  # Handle low-confidence scores as neutral
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
def extract_keywords(df, column, model_name='all-mpnet-base-v2', ngram_range=(1, 3), top_n=7):
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
    
    # Customize color palette, adjust width/height if needed
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
    
    # Convert wordcloud to image and return
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

# Function to analyze topics
def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=7, ngram_range=(1, 3)):
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

# Function to print usage instructions
def usage():
    print("Usage: python script.py [csv_file]")
    print("If no CSV file is provided, the default statewide_facultystaff_24.csv' will be used.")

if __name__ == "__main__":
    # Check if a file was provided as a command-line argument
    if len(sys.argv) != 2:
        print("Error: See usage.")
        usage()  
        sys.exit(1)  

    # Print usage if no arguments are provided, or use the default file
    if len(sys.argv) == 2:
        csv_file = sys.argv[1]  
    else:
        csv_file = "statewide_facultystaff_24.csv"  

    # Load the dataset dynamically
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
        progress_logger.info(f"Loading data from {csv_file}...")
    except FileNotFoundError:
        progress_logger.error(f"File {csv_file} not found. Please check the file path.")
        sys.exit(1)  

    # Question-to-column mapping
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "Q86",
        "What could your college or university do to address food and housing insecurity?": "Q87",
        "Is there anything else you would like to share?": "Q88",
        "Please select the reasons for not visiting the campus food pantry.": "Q28",
        "What are your thoughts about food availability on your campus?": "Q32",
        "Please share why you feel unsafe?": "Q44",
        "Please explain why it is difficult to find housing either on-campus or off-campus?": "Q49"
    }

    # Preprocess and analyze each question
    for question, col in question_mapping.items():
        df[col] = df[col].apply(preprocess_text)  # Preprocess text
        analyze_sentiment(df, col, question)  # Sentiment analysis
        extract_keywords(df, col)  # Keyword extraction

        # Only analyze topics if there are valid responses
        if not df[col].dropna().empty:
            analyze_topics(df, col)  # Topic modeling
        else:
            progress_logger.warning(f"No valid responses for column: {col}. Skipping topic analysis.")
