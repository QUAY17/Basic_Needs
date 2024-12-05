import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import contractions
import re
from collections import Counter, defaultdict
from wordcloud import WordCloud
import json
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import base64
from io import BytesIO
from datetime import datetime

def preprocess_text(text, stopword_config):
    """
    Unified preprocessing for text cleaning while preserving readability
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
        
    if text.lower().strip() in stopword_config['non_informative_responses']:
        return None
        
    na_patterns = ['n/a', 'na', 'not applicable', 'none', 'no answer', 
                  'prefer not to answer', 'prefer not to say']
    if any(pattern in text.lower() for pattern in na_patterns):
        return None
        
    # Clean text while preserving structure
    text = text.strip()
    text = contractions.fix(text)
    text = ' '.join(text.split())  # Remove extra whitespace
    text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)  # Keep punctuation
    
    if not text or text.isspace():
        return None
        
    return text

def get_survey_specific_stopwords():
    """Get unified stopwords configuration"""
    non_informative_responses = {
        # Uncertainty responses
        "unsure", "not sure", "i don't know", "i'm not sure", "don't know",
        "i am not sure", "i'm not too sure", "i am not knowledgeable about this",
        "i have no idea",
        
        # Empty/negative responses
        "nothing", "no", "none", "prefer not to say", "no comment",
        "sorry no ideas", "i have no comment", "i am unsure",
        "i have no suggestions", "not at this time", "no thank you"
    }

    stopwords = set(ENGLISH_STOP_WORDS)
    
    return {
        'stopwords': stopwords,
        'non_informative_responses': list(non_informative_responses)
    }

def prepare_data(df: pd.DataFrame, 
                question_col: str,
                stopword_config: dict,
                test_size: float = 0.2,
                val_size: float = 0.1) -> tuple:
    """Prepare and split data for ML pipeline"""
    # Convert to string first and then preprocess
    texts = df[question_col].astype(str).apply(lambda x: preprocess_text(x, stopword_config))
    valid_texts = texts.dropna()
    
    # Add minimum length check
    valid_texts = valid_texts[valid_texts.str.split().str.len() >= 3]
    
    if len(valid_texts) < 50:  # Minimum threshold for meaningful analysis
        raise ValueError(f"Not enough valid responses for {question_col} after preprocessing")
    
    train_val, test = train_test_split(valid_texts, test_size=test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)
    
    return train, val, test


def evaluate_topic_model(model: BERTopic,
                        train_embeddings: np.ndarray,
                        val_embeddings: np.ndarray,
                        train_topics: list,
                        val_topics: list) -> dict:
    metrics = {}
    
    try:
        total_docs = len(train_topics)
        meaningful_topics = [t for t in train_topics if t != -1]
        num_topics = len(set(meaningful_topics))
        
        metrics.update({
            'total_documents': total_docs,
            'number_of_topics': num_topics,
            'percent_in_topics': len(meaningful_topics) / total_docs * 100,
            'percent_outliers': (total_docs - len(meaningful_topics)) / total_docs * 100
        })
        
        topic_sizes = Counter(train_topics)
        if -1 in topic_sizes:
            del topic_sizes[-1]
            
        if topic_sizes:
            sizes = list(topic_sizes.values())
            metrics.update({
                'max_topic_size': max(sizes),
                'min_topic_size': min(sizes),
                'avg_topic_size': sum(sizes) / len(sizes),
                'topic_size_std': np.std(sizes),
                'size_distribution': {str(k): v for k, v in sorted(topic_sizes.items())}  # Convert keys to strings
            })
            
        try:
            topic_info = model.get_topic_info()
            metrics['coherence'] = float(topic_info['Coherence'].mean())
            
            similarities = model.topic_similarities(topics=list(topic_sizes.keys()))
            avg_similarity = np.mean([s for s in similarities.flatten() if s < 1.0])
            metrics['avg_topic_similarity'] = float(avg_similarity)
            
        except Exception as e:
            print(f"Warning: Could not calculate some quality metrics: {e}")
        
    except Exception as e:
        print(f"Error in topic model evaluation: {e}")
    
    return metrics

def train_topic_model(train_texts: pd.Series,
                     val_texts: pd.Series,
                     model_name: str = 'all-mpnet-base-v2') -> tuple:
    """
    Train topic model with adjusted parameters for better topic discovery
    """
    n_responses = len(train_texts)
    
    # More lenient minimum topic size
    min_topic_size = max(5, int(n_responses * 0.02))  # 2% of responses or at least 5
    
    embedding_model = SentenceTransformer(model_name)
    stopwords = list(get_survey_specific_stopwords()['stopwords'])
    
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),  # Reduced from (1,3) to catch more basic patterns
        stop_words=stopwords,
        token_pattern=r'\b\w+\b',
        min_df=1,  # Reduced from 2 to catch rare but important terms
        max_df=0.95
    )
    
    # Initialize with more lenient parameters
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer,
        min_topic_size=min_topic_size,
        n_gram_range=(1, 2),
        top_n_words=15,
        verbose=True,
        calculate_probabilities=False,  # Speed up processing
        nr_topics='auto'  # Let the model determine the optimal number of topics
    )
    
    # Fit and transform
    try:
        train_topics, _ = topic_model.fit_transform(train_texts.tolist())
        
        # Get embeddings
        train_embeddings = embedding_model.encode(train_texts.tolist())
        val_embeddings = embedding_model.encode(val_texts.tolist())
        
        # Get final topic assignments
        train_topics, _ = topic_model.transform(train_texts.tolist())
        val_topics, _ = topic_model.transform(val_texts.tolist())
        
        # Calculate metrics with error handling
        try:
            metrics = evaluate_topic_model(
                topic_model,
                train_embeddings,
                val_embeddings,
                train_topics,
                val_topics
            )
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            metrics = {
                'total_documents': len(train_topics),
                'number_of_topics': len(set(t for t in train_topics if t != -1))
            }
        
        return topic_model, train_topics, metrics
        
    except Exception as e:
        raise RuntimeError(f"Error in topic modeling: {e}")

def extract_keywords(texts: pd.Series,
                    topic_model: BERTopic,
                    topics: list,
                    question: str,
                    stopwords=None) -> list:
    """
    Extract diverse keywords and meaningful examples
    """
    if stopwords is None:
        stopwords = get_survey_specific_stopwords()['stopwords']

    topic_summary = []
    
    # Convert texts to strings if they aren't already
    texts = texts.astype(str)
    
    for topic_idx in set(topics):
        if topic_idx == -1:  # Skip outlier topic
            continue

        # Get texts for this topic
        topic_mask = np.array(topics) == topic_idx
        topic_texts = texts[topic_mask]
        
        if len(topic_texts) < 3:  # Skip very small topics
            continue
        
        try:
            # Get keywords and their scores
            topic_words = topic_model.get_topic(topic_idx)
            cleaned_keywords = {}
            
            for word, score in topic_words:
                word = str(word).lower().strip()
                if word in stopwords or len(word) < 3:
                    continue
                if not any(word in existing or existing in word 
                          for existing in cleaned_keywords.keys()):
                    cleaned_keywords[word] = int(score * 100)

            # Select diverse examples
            examples = []
            sorted_responses = sorted(topic_texts.astype(str), key=len, reverse=True)
            
            for response in sorted_responses:
                if len(response.split()) < 3:
                    continue
                    
                # Skip if too similar to existing examples
                content_words = set(response.lower().split())
                if any(len(content_words.intersection(set(ex.lower().split()))) / len(content_words) > 0.7 
                      for ex in examples):
                    continue
                    
                examples.append(response)
                if len(examples) >= 5:
                    break

            if examples:  # Only add topics with valid examples
                topic_summary.append({
                    'Topic': topic_idx,
                    'Count': len(topic_texts),
                    'Examples': examples[:5],
                    'Keywords': list(cleaned_keywords.keys())[:10],
                    'KeywordCounts': cleaned_keywords
                })

        except Exception as e:
            print(f"Warning: Error processing topic {topic_idx}: {e}")
            continue

    # Sort topics by size
    topic_summary.sort(key=lambda x: x['Count'], reverse=True)
    
    return topic_summary

def analyze_student_responses(file_paths: dict,
                            question_mapping: dict,
                            stopword_config: dict,
                            ) -> dict:
    """
    Enhanced analysis pipeline handling both qualitative and categorical questions
    """
    results = {}
    
    for group, filepath in file_paths.items():
        try:
            df = pd.read_csv(filepath, low_memory=False)
            group_results = {'qualitative': {}, 'categorical': {}}
            
            # Process each question based on its type
            for question, info in question_mapping.items():
                if isinstance(info, str):  # Qualitative/open-ended question
                    column = info
                    # Prepare data
                    train_texts, val_texts, test_texts = prepare_data(
                        df, column, stopword_config)
                    
                    if len(train_texts) < 50:
                        print(f"Insufficient responses for {column} in {group}")
                        continue
                    
                    # Train topic model
                    topic_model, topics, metrics = train_topic_model(
                        train_texts,
                        val_texts
                    )
                    
                    # Extract keywords
                    topic_summary = extract_keywords(
                        train_texts,
                        topic_model,
                        topics,
                        question,
                        stopwords=stopword_config['stopwords']
                    )
                    
                    # Store qualitative results
                    group_results['qualitative'][question] = {
                        'metrics': metrics,
                        'topic_summary': topic_summary,
                        'response_stats': {
                            'total_responses': len(df),
                            'valid_responses': len(train_texts),
                            'response_rate': (len(train_texts) / len(df)) * 100
                        }
                    }
                    
                elif isinstance(info, dict):  # Categorical question
                    column = info['question']
                    question_type = info['type']
                    
                    if question_type == 'categorical_single':
                        analysis = analyze_single_categorical(df, column, info)
                    elif question_type == 'categorical_multiple':
                        analysis = analyze_multiple_categorical(df, column, info)
                    
                    group_results['categorical'][question] = {
                        'analysis': analysis,
                        'metadata': info
                    }
            
            results[group] = group_results
            
        except Exception as e:
            print(f"Error analyzing {group}: {str(e)}")
            continue
    
    return results

def analyze_single_categorical(df: pd.DataFrame, column: str, info: dict) -> dict:
    """Analyze single-select categorical questions"""
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
    
    analysis['response_rate'] = (analysis['valid_responses'] / analysis['total_responses']) * 100
    return analysis

def analyze_multiple_categorical(df: pd.DataFrame, column: str, info: dict) -> dict:
    """Analyze multiple-select categorical questions"""
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
    
    analysis['response_rate'] = (analysis['valid_responses'] / analysis['total_eligible']) * 100
    return analysis

def generate_wordcloud_from_keywords(topic_summary, column, question, output_dir='.', stopwords=None, max_words=30) -> tuple:
    """Generate a wordcloud from topic modeling results"""
    if not topic_summary:
        print(f"No topics found for column: {column}")
        return None, None

    word_freq = defaultdict(float)
    total_responses = sum(topic['Count'] for topic in topic_summary)

    for topic in topic_summary:
        topic_size = topic['Count']
        topic_weight = topic_size / total_responses

        for keyword, count in topic['KeywordCounts'].items():
            keyword = keyword.lower().strip()
            
            if stopwords and keyword in stopwords:
                continue
                
            word_freq[keyword] += count * topic_weight

    if not word_freq:
        print(f"No valid keywords for column: {column}")
        return None, None

    max_freq = max(word_freq.values())
    word_freq = {word: (freq/max_freq) * 100 for word, freq in word_freq.items()}

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"wordcloud_{column}_{question[:30]}.jpg"
    output_path = os.path.join(output_dir, output_filename)

    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        color_func=custom_teal_color_func,
        max_words=max_words,
        relative_scaling=1,
        min_font_size=8,
        max_font_size=120,
        collocations=False
    ).generate_from_frequencies(word_freq)

    wordcloud.to_file(output_path)

    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="JPEG", quality=95)
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str, output_path

def custom_teal_color_func(word: str, 
                          font_size: int, 
                          position: tuple, 
                          orientation: int, 
                          random_state: int = None, 
                          **kwargs) -> str:
    """Custom color function for consistent teal coloring in wordcloud"""
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

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}  # Convert all keys to strings
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
    # Initialize configurations
    stopword_config = get_survey_specific_stopwords()
    
    file_paths = {
        'undergrad_ft': 'data/student/undergrad_ft_data.csv',
    }
    
    try:
        # Get question mappings
        question_mapping = get_question_mapping()
        
        # Run combined analysis
        results = analyze_student_responses(
            file_paths, 
            question_mapping,
            stopword_config,
        )
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_results = {
            'metadata': {
                'timestamp': timestamp,
                'num_groups': len(file_paths),
                'num_questions': len(question_mapping)
            },
            'analysis_results': convert_numpy_types(results)
        }
        
        os.makedirs('results', exist_ok=True)
        results_file = os.path.join('results', f'analysis_results_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=4)
            
        print(f"Analysis results saved to {results_file}")
            
    except Exception as e:
        print(f"Error in main analysis pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    
    main()