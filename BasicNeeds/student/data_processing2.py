import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.feature_extraction.text import CountVectorizer
import mlflow
import logging
import contractions
import re
from collections import Counter, defaultdict
from wordcloud import WordCloud
import json
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import base64
from io import BytesIO

def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Handlers setup as in original code
    file_handler = logging.FileHandler("analysis.log")
    stream_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Usage in preprocessing
def preprocess_text(text, stopword_config):
    """
    Preprocess survey text with configured stopwords
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
        
    # Check non-informative responses FIRST
    if text.lower().strip() in stopword_config['non_informative_responses']:
        return None
        
    # Add NA variations to filter
    na_patterns = ['n/a', 'na', 'not applicable', 'none', 'no answer', 
                  'prefer not to answer', 'prefer not to say']
    if any(pattern in text.lower() for pattern in na_patterns):
        return None
        
    # Clean text
    text = text.lower().strip()
    text = contractions.fix(text)
    
    # Remove non-word characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove single characters and numbers
    words = [w for w in text.split() if len(w) > 1]
    
    # Filter stopwords
    words = [w for w in words if w not in stopword_config['stopwords']]
    
    # If nothing meaningful left, return None
    if not words:
        return None
        
    return ' '.join(words)

def get_survey_specific_stopwords():
    """
    Get stopwords specifically tailored for this basic needs survey,
    preserving important negative and frequency words
    """

    # Non-informative complete responses to filter out
    non_informative_responses = {
        # Uncertainty responses
        "unsure", "not sure", "i don't know", "im not sure", 
        "dont know", "i am not sure", "i'm not too sure",
        "i am not knowledgeable about this", "i have no idea","la", "el", "los", "de", "son"
        
        # Empty/negative responses
        "nothing", "no", "none", "prefer not to say", "no comment",
        "sorry no ideas", "i have no comment", "i am unsure",
        "i have no suggestions", "not at this time", "no thank you", "dont", "don't"
    }

    # Start with sklearn's English stopwords
    stopwords = set(ENGLISH_STOP_WORDS)

    return {
        'stopwords': stopwords,
        'non_informative_responses': list(non_informative_responses)
    }


def prepare_data(df: pd.DataFrame, 
                question_col: str,
                logger,
                stopword_config: dict,
                test_size=0.2,
                val_size=0.1) -> tuple:
    """
    Prepare and split data for ML pipeline
    """
    # Preprocess texts
    # Preprocess texts with stopwords
    texts = df[question_col].apply(lambda x: preprocess_text(x, stopword_config))
    valid_texts = texts.dropna()
    
    # Split data
    train_val, test = train_test_split(valid_texts, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    logger.info(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return train, val, test

def train_topic_model(train_texts: pd.Series,
                     val_texts: pd.Series,
                     logger,
                     min_topic_size=7,
                     model_name='all-mpnet-base-v2') -> tuple:
    """
    Train and evaluate topic model with proper ML practices
    """
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model_name": model_name,
            "min_topic_size": min_topic_size
        })
        
        # Initialize models
        embedding_model = SentenceTransformer(model_name)

        # Convert stopwords set to list for CountVectorizer
        stopwords = list(get_survey_specific_stopwords()['stopwords'])
        
        # Initialize vectorizer
        vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            stop_words=stopwords,
            token_pattern=r'\b\w+\b',
            min_df=2
        )
        
        # Initialize topic model
        topic_model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=vectorizer,
            min_topic_size=min_topic_size
        )
        
        # Fit model on training data
        train_topics, _ = topic_model.fit_transform(train_texts.tolist())
        
        # Get embeddings for evaluation
        train_embeddings = embedding_model.encode(train_texts.tolist())
        val_embeddings = embedding_model.encode(val_texts.tolist())
        
        # Evaluate on validation set
        val_topics, _ = topic_model.transform(val_texts.tolist())
        
        # Calculate metrics
        metrics = evaluate_topic_model(
            topic_model, 
            train_embeddings, 
            val_embeddings,
            train_topics,
            val_topics
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        return topic_model, train_topics, metrics

def evaluate_topic_model(model, train_embeddings, val_embeddings, 
                        train_topics, val_topics) -> dict:
    """
    Simple evaluation focusing on topic distribution and coverage
    """
    metrics = {}
    
    try:
        # Basic topic distribution stats
        total_docs = len(train_topics)
        meaningful_topics = [t for t in train_topics if t != -1]  # Exclude outlier topic (-1)
        num_topics = len(set(meaningful_topics))
        
        metrics.update({
            'total_documents': total_docs,
            'number_of_topics': num_topics,
            'percent_in_topics': len(meaningful_topics) / total_docs * 100,  # % docs in meaningful topics
            'percent_outliers': (total_docs - len(meaningful_topics)) / total_docs * 100  # % docs in outlier topic
        })
        
        # Topic size distribution
        topic_sizes = Counter(train_topics)
        if -1 in topic_sizes:
            del topic_sizes[-1]
            
        if topic_sizes:
            sizes = list(topic_sizes.values())
            metrics.update({
                'max_topic_size': max(sizes),
                'min_topic_size': min(sizes),
                'avg_topic_size': sum(sizes) / len(sizes)
            })
            
    except Exception as e:
        logger.warning(f"Error calculating topic metrics: {e}")
    
    return metrics

def analyze_student_responses(file_paths: dict,
                            question_mapping: dict,
                            stopword_config: dict,
                            logger) -> dict:
    """
    Main analysis pipeline with ML best practices
    """
    # Get stopwords configuration once
    stopwords = get_survey_specific_stopwords()['stopwords']
    
    results = {}
    
    for group, filepath in file_paths.items():
        logger.info(f"Analyzing {group} responses...")
        
        # Load data
        df = pd.read_csv(filepath, low_memory=False)
        
        group_results = {}
        for question, col in question_mapping.items():
            # Prepare data
            train_texts, val_texts, test_texts = prepare_data(df, col, logger, stopword_config)
            
            # Train and evaluate topic model
            topic_model, topics, metrics = train_topic_model(
                train_texts,
                val_texts,
                logger
            )
            
            # Extract keywords with stopwords
            topic_summary = extract_keywords(
                train_texts,
                topic_model,
                topics,
                question,
                logger,
                stopwords=stopword_config['stopwords']  # Pass stopwords here
            )
            
            # Generate visualizations using original wordcloud function
            wordcloud_img, wordcloud_path = generate_wordcloud_from_keywords(
                topic_summary,
                col,
                question,
                output_dir=f"wordclouds_{group}"
            )
            
            group_results[question] = {
                'metrics': metrics,
                'topic_summary': topic_summary,
                'wordcloud': {
                    'image': wordcloud_img,
                    'path': wordcloud_path
                }
            }
            
        results[group] = group_results
    
    return results

def extract_keywords(texts: pd.Series,
                    topic_model: BERTopic,
                    topics: list,
                    question: str,
                    logger,
                    stopwords=None) -> dict:
    """
    Extract keywords using BERTopic and KeyBERT.
    """
    logger.info(f"Extracting keywords for question: {question}")

    # Initialize KeyBERT with stopwords handling
    kw_model = KeyBERT()

    # Get stopwords if not provided
    if stopwords is None:
        stopwords = get_survey_specific_stopwords()['stopwords']

    # Initialize topic summary
    topic_summary = []
    keyword_counts = defaultdict(int)  # To aggregate keyword frequencies globally

    for topic_idx in set(topics):
        if topic_idx == -1:
            continue

        # Get texts for this topic
        topic_texts = texts[np.array(topics) == topic_idx]

        # Get keywords from BERTopic
        bert_keywords = topic_model.get_topic(topic_idx)

        # Aggregate keywords and their counts, ensure consistent processing
        for word, score in bert_keywords:
            word = word.lower().strip()
            if word in stopwords:
                continue
            keyword_counts[word] += int(score * 100)

        logger.info(f"Keywords for topic {topic_idx}: {keyword_counts}")

        topic_summary.append({
            'Topic': topic_idx,
            'Count': len(topic_texts),
            'Examples': topic_texts[:30].tolist(),
            'Keywords': list(keyword_counts.keys()),
            'KeywordCounts': dict(keyword_counts)  # Add global count here
        })

    return topic_summary


def generate_wordcloud_from_keywords(topic_summary, column, question, output_dir='.', stopwords=None, max_words=30) -> tuple:
    """
    Generate a wordcloud from the given topic summary, emphasizing important keywords and filtering out irrelevant terms.
    """
    logger.info(f"Generating wordcloud for column: {column}...")

    # Ensure that the topic_summary is valid and has content
    if not topic_summary:
        logger.warning(f"No topics found for column: {column}. Cannot generate word cloud.")
        return None, None

    # Initialize a dictionary to hold the final keyword frequencies
    word_freq = defaultdict(int)

    # Aggregate keyword frequencies across all topics related to the column
    for topic in topic_summary:
        for keyword, count in topic['KeywordCounts'].items():
            # Convert keyword to lowercase and strip spaces to ensure consistency
            keyword = keyword.lower().strip()

            # Skip stopwords
            if stopwords and keyword in stopwords:
                continue

            # Add keyword count to the frequency dictionary
            word_freq[keyword] += count

    # If no valid keywords are found after aggregation
    if not word_freq:
        logger.warning(f"No valid keywords found for column: {column}. Cannot generate word cloud.")
        return None, None

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate a dynamic output filename based on the column name and question
    output_filename = f"wordcloud_{column}_{question[:30]}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    # Create the word cloud based on the aggregated keyword frequencies
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=custom_teal_color_func,  # Apply the custom teal color function
        max_words=max_words,
        collocations=False  # Prevent bigrams/trigrams from getting undue prominence
    ).generate_from_frequencies(word_freq)

    # Save the word cloud as a JPG image
    wordcloud.to_file(output_path)
    logger.info(f"Wordcloud saved as {output_path}")

    # Convert wordcloud to image and return base64-encoded string for embedding
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    # Return the image as base64-encoded string (for embedding in HTML) and the file path
    return img_str, output_path


def custom_teal_color_func(word: str, 
                          font_size: int, 
                          position: tuple, 
                          orientation: int, 
                          random_state: int = None, 
                          **kwargs) -> str:
    """
    Custom color function for consistent teal coloring in wordcloud
    """
    return "#1E566C"

def get_question_mapping():
    """
    Returns mapping of survey questions to their corresponding column names with both 
    quantitative and qualitative components
    """
    return {
        # Open-ended food insecurity questions
        "Please share obstacle(s) that affects ability to access healthy food.": "Obstacles",  # Q26
        "What are your thoughts about food availability on your campus?": "Foodavail",  # Q32
        
        # Food pantry questions (categorical)
        "Have you used the campus food pantry?": {
            'question': "Foodpantry",  # Q27
            'type': 'categorical_single',
            'options': ['Yes', 'No', "My campus does not have a food pantry"]
        },
        "Please select the reasons for not visiting the campus food pantry.": {
            'question': "Foodpantry_reasons",  # Q28
            'type': 'categorical_multiple',
            'options': [
                'I do not need assistance with obtaining food and household supplies',
                'The location is inconvenient',
                'The hours of operation for the campus food pantry do not work for me',
                'I visit another food pantry/food bank in my community',
                'I do not like the selection offered by the campus food pantry',
                'The items available at the campus food pantry do not align with my dietary needs',
                'I am not sure I am eligible to use the campus food pantry',
                'Other students need this help more than I do',
                'I don\'t want other people to see me and know that I am food insecure',
                'Other'
            ],
            'dependency': {
                'question': 'Foodpantry',
                'value': 'No'
            }
        },
        
        # Safety questions (mixed)
        "How safe have you felt?": {
            'question': "Safety",  # Q42
            'type': 'categorical_single',
            'options': ['Very safe', 'Moderately safe', 'Not very safe', 'Not at all safe']
        },
        "Why do you feel unsafe?": "Unsafe_why",  # Q44
        
        # Housing questions (mixed)
        "Was it difficult to find housing on or near campus?": {
            'question': "Housingdiff",  # Q48
            'type': 'categorical_single',
            'options': ['Yes', 'No', 'Not applicable']
        },
        "Please explain why it was difficult to find housing.": "Housingdiff_why",  # Q49
        
        # Impact and suggestions (open-ended)
        "Please share a personal experience where food or housing insecurity had a direct impact.": "Experience",  # Q52
        "What could your college or university do to address food and housing insecurity?": "OE2",  # Q87
        "Is there anything else you would like to share?": "OE3"  # Q88
    }

def analyze_categorical_responses(file_paths: dict,
                                categorical_questions: dict,
                                logger: logging.Logger) -> dict:
    """
    Analyze categorical responses across different student groups
    
    Args:
        file_paths: Dictionary mapping student groups to their data files
        categorical_questions: Dictionary of categorical questions with their metadata
        logger: Logger instance
    
    Returns:
        Dictionary containing analysis results per group and question
    """
    results = {}
    
    for group, filepath in file_paths.items():
        logger.info(f"Analyzing categorical responses for {group}")
        
        # Load data for this group
        try:
            df = pd.read_csv(filepath, low_memory=False)
            group_results = {}
            
            for question, info in categorical_questions.items():
                column = info['question']
                question_type = info['type']
                
                # Skip if column not in dataframe
                if column not in df.columns:
                    logger.warning(f"Column {column} not found in {group} data")
                    continue
                
                # Analyze based on question type
                if question_type == 'categorical_single':
                    analysis = {
                        'counts': df[column].value_counts().to_dict(),
                        'percentages': (df[column].value_counts(normalize=True) * 100).to_dict(),
                        'missing': df[column].isna().sum(),
                        'total_responses': len(df),
                        'valid_responses': df[column].notna().sum(),
                        'unique_values': list(df[column].unique()),
                        'options_found': [opt for opt in info['options'] if opt in df[column].unique()],
                        'unexpected_values': [val for val in df[column].unique() 
                                           if val not in info['options'] and pd.notna(val)]
                    }
                    
                    # Add response rate
                    analysis['response_rate'] = (analysis['valid_responses'] / analysis['total_responses']) * 100
                    
                elif question_type == 'categorical_multiple':
                    # Handle dependency if exists
                    if 'dependency' in info:
                        dep_col = info['dependency']['question']
                        dep_val = info['dependency']['value']
                        eligible_df = df[df[dep_col] == dep_val]
                    else:
                        eligible_df = df
                    
                    # Split multiple responses if stored as comma-separated
                    if eligible_df[column].dtype == 'object':
                        responses = eligible_df[column].str.split(',').explode()
                    else:
                        responses = eligible_df[column]
                    
                    analysis = {
                        'counts': responses.value_counts().to_dict(),
                        'percentages': (responses.value_counts(normalize=True) * 100).to_dict(),
                        'missing': responses.isna().sum(),
                        'total_eligible': len(eligible_df),
                        'valid_responses': responses.notna().sum(),
                        'unique_values': list(responses.unique()),
                        'options_found': [opt for opt in info['options'] if opt in responses.unique()],
                        'unexpected_values': [val for val in responses.unique() 
                                           if val not in info['options'] and pd.notna(val)]
                    }
                    
                    # Add multiple selection patterns
                    if eligible_df[column].dtype == 'object':
                        selection_counts = eligible_df[column].str.split(',').str.len().value_counts()
                        analysis['multiple_selection_patterns'] = selection_counts.to_dict()
                    
                    # Add response rate
                    analysis['response_rate'] = (analysis['valid_responses'] / analysis['total_eligible']) * 100
                
                # Add to group results
                group_results[question] = {
                    'analysis': analysis,
                    'metadata': info
                }
                
                logger.info(f"Completed analysis of {column} for {group}: "
                          f"{analysis['valid_responses']} valid responses "
                          f"({analysis['response_rate']:.1f}% response rate)")
            
            # Add group results to overall results
            results[group] = group_results
            
        except Exception as e:
            logger.error(f"Error analyzing {group}: {str(e)}")
            continue
    
    return results

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    return obj

def main():
    # Initialize stopwords configuration once
    stopword_config = get_survey_specific_stopwords()

    file_paths = {
        'undergrad_ft': 'data/student/undergrad_ft_data.csv',
        # 'undergrad_pt': 'undergrad_pt_data.csv',
        # 'graduate': 'grad_data.csv'
    }
    
    # Get and split questions
    question_mapping = get_question_mapping()
    qualitative_questions = {q: info for q, info in question_mapping.items() if isinstance(info, str)}
    categorical_questions = {q: info for q, info in question_mapping.items() 
                           if isinstance(info, dict) and info['type'] in ['categorical_single', 'categorical_multiple']}
    
    # Run analyses
    logger.info("Starting qualitative analysis...")
    # Pass to analysis functions
    qual_results = analyze_student_responses(
        file_paths, 
        qualitative_questions,
        stopword_config,  # Pass stopwords once
        logger
    )
    
    logger.info("Starting categorical analysis...")
    cat_results = analyze_categorical_responses(file_paths, categorical_questions, logger)
    
    # Save all results to one file
    final_results = {
        'metadata': {
            'num_groups': len(file_paths),
            'num_qualitative_questions': len(qualitative_questions),
            'num_categorical_questions': len(categorical_questions)
        },
        'qualitative_analysis': convert_numpy_types(qual_results),
        'categorical_analysis': convert_numpy_types(cat_results)
    }
    
    output_file = f'analysis_results.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=4)
        logger.info(f"All results successfully saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    logger = setup_logging()
    main()