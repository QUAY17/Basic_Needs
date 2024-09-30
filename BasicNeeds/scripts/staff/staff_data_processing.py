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
import json

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
        return None  # Return None for invalid or missing entries

    # Convert to lowercase
    text = text.lower()

    # Expand contractions like "doesn't" -> "does not"
    text = contractions.fix(text)

    # Remove punctuation (optional)
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Function to filter out non-informative responses
def filter_responses(text):
    non_informative_responses = [
        "unsure", "not sure", "nothing", "no", "none", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not at this time", "no thank you"
    ]
    return text not in non_informative_responses

def analyze_topics(valid_texts, column, model_name='all-mpnet-base-v2', min_topic_size=7, merge_threshold=0.7):
    progress_logger.info(f"Analyzing topics for column: {column}")
    embedding_model = SentenceTransformer(model_name)

    # Custom stopwords list (excluding negations)
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    custom_stopwords = list(ENGLISH_STOP_WORDS.difference({'no', 'not',  'nor', 'neither', 'never', 'none', 'nothing', 'nobody', 'nowhere', 'without',
    'don\'t', 'won\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'doesn\'t', 'didn\'t',
    'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'mustn\'t', 'mightn\'t', 'always', 'never', 'sometimes', 'often', 'seldom', 'rarely', 'frequently', 'usually', 'every', 'all', 'any', 'none'}))

    # Initialize vectorizer with adjusted settings
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 3),  # Capture unigrams and bigrams
        stop_words=custom_stopwords,
        token_pattern=r'\b\w+\b',  # Include words with at least one character
        min_df=2,
    )
    
    # Initialize topic model with custom parameters
    # vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=min_topic_size, vectorizer_model=vectorizer_model)

    topics, _ = topic_model.fit_transform(valid_texts.tolist())

    # Summarize topics before merging
    topic_info, topic_summary = summarize_topics(topic_model, topics, valid_texts)

    # Merge similar topics based on the threshold
    topic_summary = merge_similar_topics(topic_summary, threshold=merge_threshold)

    # Save topic info
    topic_info.to_csv(f"topic_analysis_results_{column}.csv", index=False)

    # Save topic info and summaries categorized by institution
    output_path = f"topic_summary_{column}.txt"
    with open(output_path, "w") as f:
        for topic in topic_summary:
            f.write(f"Topic {topic['Topic']} (Count: {topic['Count']})\n")
            f.write("Examples:\n")
            for response, institution in topic['Examples']:
                f.write(f"Institution: {institution} | Response: {response}\n")
            f.write("\n")
    
    progress_logger.info(f"Topic analysis completed for column: {column}. Summary saved to {output_path}")
    return topic_model, topics, topic_summary

# Function to summarize topics and count keyword occurrences
def summarize_topics(topic_model, topics, texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    # Build a set of all n-grams in the corpus
    ngram_counts = Counter()
    for text in texts:
        tokens = text.split()
        for n in range(1, 3):  # Adjust according to your n-gram range
            ngrams = zip(*[tokens[i:] for i in range(n)])
            for ngram in ngrams:
                ngram_counts[' '.join(ngram)] += 1

    for topic in topic_info['Topic']:
        if topic == -1:
            continue  # Skip outliers

        # Get all texts corresponding to the current topic
        topic_texts = [text for t, text in zip(topics, texts) if t == topic]
        examples = topic_texts[:50]  # Take the first 50 examples for reference

        # Extract keywords for this topic
        raw_keywords = topic_model.get_topic(topic)

        # Filter keywords to only include those present in the corpus
        keywords = []
        for word, _ in raw_keywords:
            if word in ngram_counts and ngram_counts[word] > 0:
                keywords.append(word)
            if len(keywords) >= 10:
                break  # Limit to top 10 keywords

        # Skip topics with no informative keywords
        if not keywords:
            continue

        # Initialize keyword counts
        keyword_counts = {kw: ngram_counts[kw] for kw in keywords}

        # Sort keywords by their counts in descending order
        sorted_keyword_counts = dict(sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True))
        sorted_keywords = list(sorted_keyword_counts.keys())

        # Append the summary for the current topic
        topic_summary.append({
            'Topic': topic,
            'Count': len(topic_texts),
            'Examples': examples,
            'Keywords': sorted_keywords,  # Sorted keywords
            'KeywordCounts': sorted_keyword_counts  # Sorted keyword counts
        })

    # Write the topic_summary to the file in JSON format for readability
    with open('output.txt', 'w') as file:
        json.dump(topic_summary, file, indent=4)
    
    exit(0)
    
    return topic_info, topic_summary

def summarize_topics_by_institution(topic_model, topics, texts, institutions):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    for topic in topic_info['Topic']:
        if topic == -1:
            continue  # Skip outliers
        
        topic_count = sum(1 for t in topics if t == topic)
        topic_texts = [(text, institution) for t, text, institution in zip(topics, texts, institutions) if t == topic]
        examples = topic_texts[:50]
        
        # Extract keywords for this topic
        keywords = [word for word, _ in topic_model.get_topic(topic)]
        
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'Examples': examples,
            'Keywords': keywords  # Include extracted keywords for merging
        })

    return topic_info, topic_summary

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
        df[col] = df[col].apply(preprocess_text)  # Preprocess text

        institutions = df['Institution'].tolist()

        # Filter out empty or invalid rows after preprocessing
        valid_texts = df[col].dropna()

        # Only analyze topics if there are valid responses
        if valid_texts.any():
            # Topic modeling
            topic_model, topics, topic_summary = analyze_topics(valid_texts, col)  # Topic modeling
            """
            if topic_model:
                # Extract keywords from the topics (post-topic generation)
                keyword_summary = extract_keywords_from_larger_topics(topic_model, topics, valid_texts.tolist())  # Keyword extraction
                
                # Generate the word cloud based on the keyword summary
                generate_wordcloud_from_keywords(keyword_summary, col)  # Use the keyword summary for word cloud generation
            """
        else:
            progress_logger.warning(f"No valid responses for column: {col}. Skipping topic and keyword analysis.")

"""

def extract_keywords_from_larger_topics(topic_model, topics, texts, min_count=10, ngram_range=(1, 5), top_n=7):
    keyword_extractor = KeyBERT(model=topic_model.embedding_model)
    keyword_results = {}
    
    for topic_id in set(topics):
        # Filter texts for larger topics
        topic_texts = [text for t, text in zip(topics, texts) if t == topic_id and isinstance(text, str)]
        
        if len(topic_texts) >= min_count:  # Only analyze topics with a minimum number of responses
            combined_text = " ".join(topic_texts)
            keywords = keyword_extractor.extract_keywords(combined_text, keyphrase_ngram_range=ngram_range, top_n=top_n)
            
            # Store the keywords for each topic
            keyword_results[topic_id] = [kw[0] for kw in keywords]

    # Aggregate keywords across all topics and count their occurrences
    keyword_counter = Counter([kw for kws in keyword_results.values() for kw in kws])
    
    # Create a summary DataFrame with keyword counts
    keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)

    # Optionally save the overall keyword summary
    keyword_summary.to_csv(f"keyword_analysis_larger_topics.csv", index=False)

    return keyword_summary

# Function to generate word cloud
def generate_wordcloud_from_keywords(keyword_summary, column, output_dir='.'):
    progress_logger.info(f"Generating wordcloud for column: {column}...")

    # Ensure that the keyword_summary DataFrame is valid and has content
    if keyword_summary.empty:
        progress_logger.warning("Empty keyword summary provided. Cannot generate word cloud.")
        return None, None

    # Convert the keywords and their frequencies into a dictionary for the word cloud
    word_freq = dict(zip(keyword_summary['Keyword'], keyword_summary['Count']))

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a dynamic output filename based on the column name
    output_filename = f"wordcloud_{column}.jpg"

    # Generate the word cloud based on the keyword frequencies
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate_from_frequencies(word_freq)

    # Construct the full path where the word cloud image will be saved
    output_path = os.path.join(output_dir, output_filename)

    # Save the word cloud as a JPG image
    wordcloud.to_file(output_path)
    progress_logger.info(f"Wordcloud saved as {output_path}")

    # Convert wordcloud to image and return base64-encoded string for embedding
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str, output_path
    """