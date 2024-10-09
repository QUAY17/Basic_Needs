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
from collections import defaultdict

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

    # Remove punctuation
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
    custom_stopwords = list(ENGLISH_STOP_WORDS.difference({'no', 'nor', 'neither', 'never', 'nothing', 'nobody', 'nowhere', 'without',
    'don\'t', 'won\'t', 'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'doesn\'t', 'didn\'t',
    'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'mustn\'t', 'mightn\'t', 'always', 'never', 'sometimes', 'often', 'seldom', 'rarely', 'frequently', 'usually', 'every', 'any'}))

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
    # topic_summary = merge_similar_topics(topic_summary, threshold=merge_threshold)

    # Save topic info
    topic_info.to_csv(f"topic_analysis_results_{column}.csv", index=False)

    # Save topic info and summaries 
    file = f"topic_summary_{column}.txt"
    with open(file, "w") as f:
        f.write(f"Question: {question}\n\n")
        json.dump(topic_summary, f, indent=4)
    
    progress_logger.info(f"Topic analysis completed for column: {column}. Summary saved to {file}")
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
    
    return topic_info, topic_summary

# Function to get thematic keywords and filters for each question
def get_thematic_keywords(question):
    thematic_keywords_map = {
        "How is food or housing insecurity affecting your work?": {
            "thematic_keywords": set([
                "hungry", "hunger", "stress", "mental health", "afford", "food costs", "housing insecurity", 
                "rent", "bills", "focus", "concentrate", "productivity", "meal skipping", "low energy", 
                "housing stress", "unable to afford food", "worry", "anxiety"
            ]),
            "filter_out_keywords": set(["work", "na", "affect", "affecting", "currently", "sometimes", "does", "no", "yes"])
        },
         "What could your college or university do to address food and housing insecurity?": {
            "thematic_keywords": set([
                "affordable meals", "raise wages", "increase salary", "lower prices", 
                "food pantry", "emergency fund", "SNAP", "food bank", "meal plans", "nutritious meals", 
                "on-campus housing", "affordable housing", "temporary housing", "housing assistance",
                "work-study", "employment contracts", "fair wages", "benefits for staff", 
                "support services", "funding", "systemic issues", "equity", "address poverty", 
                "childcare", "daycare", "cafeteria", "food trucks", "consistent hours"
            ]),
            "filter_out_keywords": set(["don\u00e2t", "donât", "unsure", "no", "think", "know", "people", "sure", "doing", "like", "good", "great", "just", "nothing"])
        },
        "Is there anything else you would like to share?": {
            "thematic_keywords": set([
                "cost of living", "inflation", "wages", "housing support", "economic inequality", "thanks",
                "thank you", "grateful", "appreciate", "survey", "help", "food", "expensive", "housing",
                "support", "resources", "food pantry", "housing assistance", "students", "survey", "need help", "help"
                "inequality", "struggling", "hardships", "unaffordable", "time", "hungry", "low income", "cost of living", "inflation", "wages", "housing support", "economic inequality",
                "thank you", "grateful", "appreciate", "survey", "help", "food", "expensive", "housing",
                "support", "resources", "food pantry", "housing assistance", "students", "survey", "need help", "action",
                "inequality", "struggling", "hardships", "unaffordable", "time", "hungry", "low income"
            ]),
            "filter_out_keywords": set([
                "no", "think", "nothing", "good", "really", "needed", "life", "share", "said", "thanks"
            ])
        },
        "Please select the reasons for not visiting the campus food pantry.": {
            "thematic_keywords": set([
                "need assistance", "obtaining food", "household supplies", "other students need", 
                "eligible", "unsure eligibility", "food insecure", "stigma", "privacy concerns", "vegetarian", 
                "vegan", "halal", "kosher", "dietary needs", "location", "inconvenient", "hours", "operation"
            ]),
            "filter_out_keywords": set([
                "students", "food", "supplies", "need", "think", "good", "sure", "really"
            ])
        },
        "What are your thoughts about food availability on your campus?": {
            "thematic_keywords": set([
                "options", "limited options", "variety", "selection", "choices",
                "cost", "expensive", "overpriced", "affordable", 
                "availability", "accessible", 
                "healthy", "healthy options", "nutritious", "unhealthy", 
                "vending machines", "cafeteria", "food trucks", "snack bar",
                "students", "staff", "faculty", "food pantry"
            ]),
            "filter_out_keywords": set([
                "think", "know", "people", "sure", "doing", "like", "good", "great", "just"
            ])
        },
        "Why do you feel unsafe?": {
            "thematic_keywords": set([
                "crime", "gun violence", "shootings", "break-ins", "robbery", "assault", "homelessness",
                "domestic violence", "abusive relationships", "verbal abuse", "physical abuse", "partner", "family issues",
                "unsafe housing", "homeless", "poor neighborhood", "unaffordable housing", "drug use", "mentally ill",
                "unstable", "rent increase", "inflation", "economic instability", "people", "apartment", "home","live"
            ]),
            "filter_out_keywords": set([
                "no"
            ])
        },
        "There are many reasons why people are food insecure. Please share an obstacle (or two) that affects your ability to access healthy food.": {
            "thematic_keywords": set([
                "cost", "expensive", "inflation", "rising prices", "food costs", "access", "availability", "grocery stores", 
                "healthy food", "fresh food", "afford", "time", "limited options", "rural", "transportation", 
                "low income", "wages", "financial", "produce", "meat", "vegetables", "supply chain", "food desert", 
                "distance", "affordability", "budget", "choices", "grocery", "convenience", "unavailable", "income"
            ]),
            "filter_out_keywords": set([
                "no", "none", "na", "think", "good", "sure", "people", "food", "like", "get", "just"
            ])
        }
    }
    
    # Return the thematic and filter-out keywords for the specific question
    return thematic_keywords_map.get(question, {"thematic_keywords": set(), "filter_out_keywords": set()})

def custom_teal_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    teal_color = "#1E566C"
    return teal_color  # Use the specified teal color

# Function to generate a word cloud for a question, dynamically selecting thematic keywords
def generate_wordcloud_from_keywords(topic_summary, column, question, output_dir='.', max_words=20, colormap='greys'):
    progress_logger.info(f"Generating wordcloud for column: {column}...")

    # Ensure that the topic_summary is valid and has content
    if not topic_summary:
        progress_logger.warning(f"No topics found for column: {column}. Cannot generate word cloud.")
        return None, None

    # Initialize a dictionary to hold the final keyword frequencies
    word_freq = defaultdict(int)

    # Get thematic and filter-out keywords from the function based on the question
    thematic_data = get_thematic_keywords(question)
    thematic_keywords = thematic_data["thematic_keywords"]
    filter_out_keywords = thematic_data["filter_out_keywords"]

    # Aggregate keyword frequencies across all topics related to the column
    for topic in topic_summary:
        for keyword, count in topic['KeywordCounts'].items():
            
            # Filter out generic keywords
            if keyword in filter_out_keywords:
                continue
            
            # Prioritize thematic keywords and boost their frequency
            if keyword in thematic_keywords:
                word_freq[keyword] += count * 2  # Boost frequency for strong thematic keywords
            
            # Add all other keywords
            else:
                word_freq[keyword] += count

    # If no keywords found, skip
    if not word_freq:
        progress_logger.warning(f"No valid keywords found for column: {column}. Cannot generate word cloud.")
        return None, None

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a dynamic output filename based on the column name and question
    output_filename = f"wordcloud_{column}_{question}.jpg"

    # Generate the word cloud based on the aggregated keyword frequencies
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        color_func=custom_teal_color_func, 
        max_words=max_words
    ).generate_from_frequencies(word_freq)

    # Construct the full path where the word cloud image will be saved
    output_path = os.path.join(output_dir, output_filename)

    # Save the word cloud as a JPG image
    wordcloud.to_file(output_path)
    progress_logger.info(f"Wordcloud saved as {output_path}")

    # Convert wordcloud to image and return base64-encoded string for embedding
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Return the image as base64-encoded string (for embedding in HTML) and the file path
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
        # "How is food or housing insecurity affecting your work?": "OE1",
        # "What could your college or university do to address food and housing insecurity?": "OE2",
        # "Is there anything else you would like to share?": "OE3",
        # "Please select the reasons for not visiting the campus food pantry.": "Foodpantry_reasons",
        # "What are your thoughts about food availability on your campus?": "Foodavail",
        # "Please share why you feel unsafe?": "Unsafe_why",
        # "Please explain why it is difficult to find housing either on-campus or off-campus?": "Housingdiff_why",
        "There are many reasons why people are food insecure. Please share an obstacle (or two) that affects your ability to access healthy food.": "Obstacles"
    }

    # Preprocess and analyze each question
    for question, col in question_mapping.items():
        df[col] = df[col].apply(preprocess_text)  # Preprocess text

        # institutions = df['Institution'].tolist()

        # Filter out empty or invalid rows after preprocessing
        valid_texts = df[col].dropna()

        # Only analyze topics if there are valid responses
        if valid_texts.any():
            # Topic modeling
            topic_model, topics, topic_summary = analyze_topics(valid_texts, col)  # Topic modeling

            img_str, wordcloud_path = generate_wordcloud_from_keywords(topic_summary, col, question, output_dir="wordclouds")

        else:
            progress_logger.warning(f"No valid responses for column: {col}. Skipping topic and keyword analysis.") 