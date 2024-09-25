import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer
import os
import logging
import sys
import contractions
import re
from sklearn.metrics.pairwise import cosine_similarity
import spacy

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
    
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Expand contractions like "doesn't" -> "does not"
    text = contractions.fix(text)

    # Handle negations: Join 'does not', 'is not', etc. into single tokens
    text = re.sub(r'\b(does|is|has|had|was|were|could|should|would|will|shall|may|might)\s+not\b', r'\1_NOT', text)
    

    return text

# Function to filter out non-informative responses
def filter_responses(text):
    non_informative_responses = [
        "unsure", "not sure", "nothing", "no", "none", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not at this time", "no thank you"
    ]
    return text not in non_informative_responses

def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=7, ngram_range=(1, 3)):
    progress_logger.info(f"Analyzing topics for column: {column}")
    embedding_model = SentenceTransformer(model_name)
    
    # Preprocess and filter out non-informative responses
    texts = df[column].dropna().apply(preprocess_text)
    texts = texts[texts.apply(filter_responses)].tolist()
    
    if not texts:  # Check if there are no responses
        progress_logger.warning(f"No valid responses for column: {column}. Skipping topic analysis.")
        return None, None
    
    # Initialize topic model with custom parameters
    vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size, vectorizer_model=vectorizer_model)
    topics, _ = topic_model.fit_transform(texts)

    # Summarize and save topics
    topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
    
    # Save topic info and summaries
    topic_info.to_csv(f"topic_analysis_results_{column}.csv", index=False)
    with open(f"topic_summary_{column}.txt", "w") as f:
        for topic in topic_summary:
            f.write(f"Topic {topic['Topic']} (Count: {topic['Count']})\n")
            f.write("Examples:\n")
            for example in topic['Examples']:
                f.write(f"- {example}\n")
            f.write("\n")
    
    progress_logger.info(f"Topic analysis completed for column: {column}")
    return topics, topic_summary

def merge_similar_topics(topic_summary, threshold=0.7):
    # We can compute similarity between topics based on their keyword distributions and merge them
    for i, topic_a in enumerate(topic_summary):
        for j, topic_b in enumerate(topic_summary):
            if i != j:
                # Compute similarity between topic keywords
                common_keywords = set(topic_a['Keywords']).intersection(set(topic_b['Keywords']))
                similarity = len(common_keywords) / max(len(topic_a['Keywords']), len(topic_b['Keywords']))
                
                if similarity > threshold:
                    # Log the merging process
                    progress_logger.info(f"Merging topics {i} and {j} with similarity score: {similarity:.2f}")
                    
                    # Merge topics
                    topic_summary[i]['Count'] += topic_summary[j]['Count']
                    topic_summary[i]['Examples'].extend(topic_summary[j]['Examples'])
                    topic_summary[j] = None  # Mark as merged
    # Return filtered topic_summary without None
    return [topic for topic in topic_summary if topic]

# Function to summarize topics
def summarize_topics(topic_model, topics, texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    for topic in topic_info['Topic']:
        if topic == -1:
            continue
        topic_count = sum(1 for t in topics if t == topic)
        topic_texts = [text for t, text in zip(topics, texts) if t == topic]
        examples = topic_texts[:10]
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'Examples': examples
        })
    
    return topic_info, topic_summary

def extract_keywords_from_topics(topic_model, topics, texts, ngram_range=(1, 5), top_n=7):
    # Get the most frequent keywords for each topic
    keyword_extractor = KeyBERT(model=topic_model.embedding_model)
    keyword_results = {}
    
    for topic_id in set(topics):
        topic_texts = [text for t, text in zip(topics, texts) if t == topic_id]
        topic_keywords = []
        
        for text in topic_texts:
            keywords = keyword_extractor.extract_keywords(text, keyphrase_ngram_range=ngram_range, top_n=top_n)
            topic_keywords.extend([kw[0] for kw in keywords])

        # Count the keywords within this topic
        keyword_counter = Counter(topic_keywords)
        keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
        
        # Save the keyword summary for this topic
        keyword_summary.to_csv(f"keyword_analysis_topic_{topic_id}.csv", index=False)
        keyword_results[topic_id] = keyword_summary
    
    return keyword_results

# Function to count keywords
def count_keywords(keyword_results):
    keyword_counter = Counter()
    
    # Loop through the keyword results from the topic extraction
    for keywords in keyword_results:
        if isinstance(keywords, list):
            keyword_counter.update([kw[0] if isinstance(kw, tuple) else kw for kw in keywords])
        else:
            keyword_counter.update([keywords])  # In case it's a single keyword
    
    # Create a sorted DataFrame with keyword counts
    keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
    
    # Save the keyword summary as CSV (optional, if needed for reporting)
    keyword_summary.to_csv(f"keyword_analysis_summary.csv", index=False)
    
    return keyword_summary

# Function to generate word cloud
def generate_wordcloud(text, column, output_dir='.'):
    progress_logger.info(f"Generating wordcloud for column: {column}...")

    # Ensure the text is a single string (in case it's tokenized)
    if isinstance(text, list):
        text = " ".join(text)  # Join list of words into a single string

    if not text.strip():  # Ensure the text is not empty
        progress_logger.warning("Empty text provided. Cannot generate word cloud.")
        return None, None

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a dynamic output filename based on the column name
    output_filename = f"wordcloud_{column}.jpg"
    
    # Generate the word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(text)
    
    # Construct the full path where the word cloud image will be saved
    output_path = os.path.join(output_dir, output_filename)

    # Save the wordcloud as a JPG image
    wordcloud.to_file(output_path)
    progress_logger.info(f"Wordcloud saved as {output_path}")

    # Convert wordcloud to image and return base64-encoded string for embedding
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str, output_path

# Function to print usage instructions
def usage():
    print("Usage: python script.py [csv_file]")

if __name__ == "__main__":
    # Check if a file was provided as a command-line argument
    if len(sys.argv) != 2:
        print("Error: See usage.")
        usage()  
        sys.exit(1)  

    # Print usage if no arguments are provided, or use the default file
    if len(sys.argv) == 2:
        csv_file = sys.argv[1]   

    # Load the dataset dynamically
    try:
        df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
        progress_logger.info(f"Loading data from {csv_file}...")
    except FileNotFoundError:
        progress_logger.error(f"File {csv_file} not found. Please check the file path.")
        sys.exit(1)  

    # Question-to-column mapping
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        # "What could your college or university do to address food and housing insecurity?": "OE2",
        # "Is there anything else you would like to share?": "OE3",
        # "Please select the reasons for not visiting the campus food pantry.": "Foodpantry_reasons",
        # "What are your thoughts about food availability on your campus?": "Foodavail",
        # "Please share why you feel unsafe?": "Unsafe_why",
        # "Please explain why it is difficult to find housing either on-campus or off-campus?": "Housingdiff_why"
    }

    # Preprocess and analyze each question
    for question, col in question_mapping.items():
        # Preprocess the text
        df[col] = df[col].apply(preprocess_text)  

        # Only analyze topics if there are valid responses
        if not df[col].dropna().empty:
            # First, analyze topics
            topics, topic_summary = analyze_topics(df, col)  # Topic modeling
            
            # Then extract keywords from those topics
            keyword_results = extract_keywords_from_topics(topics, topic_summary, df[col].tolist())  # Keyword extraction
            
            # Count the keywords from the extracted topics
            keyword_summary = count_keywords(keyword_results)
            
            # Generate the word cloud based on the text from the column
            generate_wordcloud(" ".join(df[col].dropna()), col)  # Join all preprocessed text for word cloud generation
        else:
            progress_logger.warning(f"No valid responses for column: {col}. Skipping topic and keyword analysis.")


"""
# Function to extract keywords
def extract_keywords(df, column, model_name='all-mpnet-base-v2', ngram_range=(1, 5), top_n=7):
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
        wordcloud_img = generate_wordcloud(keywords_text, column)
    else:
        progress_logger.warning(f"No keywords found for column: {column}. Skipping word cloud generation.")

    progress_logger.info(f"Keyword extraction completed for column: {column}")

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
"""