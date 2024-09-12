import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from keybert import KeyBERT
from transformers import pipeline
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import os
from umap import UMAP

# Set environment variable to disable parallelism warning from tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower().strip()
    return text

def is_non_informative(text):
    if pd.isna(text) or not isinstance(text, str):
        return True
    non_informative_responses = [
        "unsure", "not sure", "nothing", "no", "none", "n/a", "N/A", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not applicable", "not at this time", "no thank you", "na"
    ]
    return any(response in text.lower() for response in non_informative_responses)

def measure_non_informative_responses(df, column, institution_column):
    grouped = df.groupby(institution_column)[column].apply(lambda x: x.apply(is_non_informative).sum()).reset_index()
    grouped.columns = [institution_column, 'non_informative_count']
    grouped['total_responses'] = df.groupby(institution_column)[column].count().values
    grouped['non_informative_ratio'] = grouped['non_informative_count'] / grouped['total_responses']
    return grouped

def classify_sentiment(text, sentiment, score, question, threshold=0.6):
    if is_non_informative(text):
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

def analyze_sentiment_by_institution(df, column, question, institution_column):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    sentiments = []
    for _, row in df.dropna(subset=[column]).iterrows():
        result = sentiment_pipeline(row[column])[0]
        sentiment, score = classify_sentiment(row[column], result['label'], result['score'], question)
        sentiments.append({
            'ID': row['ID'],
            'Institution': row[institution_column],
            'text': row[column],
            'label': sentiment,
            'score': score
        })
    sentiment_results = pd.DataFrame(sentiments)
    return sentiment_results

def extract_keywords_by_institution(df, column, institution_column, model_name='all-mpnet-base-v2', ngram_range=(1, 2), top_n=5):
    embedding_model = SentenceTransformer(model_name)
    keyword_extractor = KeyBERT(model=embedding_model)
    keyword_results = {}
    for institution in df[institution_column].unique():
        institution_texts = df[df[institution_column] == institution][column].dropna().apply(preprocess_text)
        institution_texts = institution_texts[~institution_texts.apply(is_non_informative)].tolist()
        if institution_texts:
            institution_keywords = [keyword_extractor.extract_keywords(text, keyphrase_ngram_range=ngram_range, stop_words='english', top_n=top_n) for text in institution_texts]
            keyword_results[institution] = institution_keywords
    return keyword_results

def count_keywords_by_institution(keyword_results):
    institution_keyword_summaries = {}
    for institution, keywords in keyword_results.items():
        keyword_counter = Counter()
        for kw_list in keywords:
            keyword_counter.update([keyword for keyword, _ in kw_list])
        keyword_summary = pd.DataFrame(keyword_counter.items(), columns=['Keyword', 'Count']).sort_values(by='Count', ascending=False)
        institution_keyword_summaries[institution] = keyword_summary
    return institution_keyword_summaries

def analyze_topics_by_institution(df, column, institution_column, model_name='all-mpnet-base-v2', min_topic_size=5, ngram_range=(1, 2), min_samples=10):
    embedding_model = SentenceTransformer(model_name)
    vectorizer_model = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    umap_model = UMAP(n_neighbors=2, n_components=2, min_dist=0.0, metric='cosine', random_state=42)

    topic_results = {}
    skipped_institutions = []
    for institution in df[institution_column].unique():
        institution_texts = df[df[institution_column] == institution][column].dropna().apply(preprocess_text)
        institution_texts = institution_texts[~institution_texts.apply(is_non_informative)].tolist()

        if len(institution_texts) >= min_samples:
            try:
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    min_topic_size=min(min_topic_size, max(2, len(institution_texts) // 5)),
                    vectorizer_model=vectorizer_model,
                    umap_model=umap_model,
                    verbose=False
                )
                topics, _ = topic_model.fit_transform(institution_texts)
                top_topics = topic_model.get_topic_info().head(3)
                topic_results[institution] = '\n'.join([f"Topic {row['Topic']}: {row['Name']}" for _, row in top_topics.iterrows()])
            except Exception as e:
                print(f"Error occurred while processing institution {institution}: {e}")
                topic_results[institution] = "Error in topic modeling"
        else:
            skipped_institutions.append(f"{institution} (Informative responses: {len(institution_texts)})")
            topic_results[institution] = f"Insufficient samples ({len(institution_texts)} < {min_samples})"

    return topic_results, skipped_institutions

def summarize_topics_by_institution(topic_results):
    institution_topic_summaries = {}
    for institution, (topic_model, topics, texts) in topic_results.items():
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
        institution_topic_summaries[institution] = (topic_info, topic_summary)
    return institution_topic_summaries

def count_responses_per_institution(df, column, institution_column):
    # Count all potential responses (including empty ones)
    total_potential_responses = df.groupby(institution_column).size()
    
    # Count non-empty responses (excluding null/NaN and empty strings)
    non_empty_responses = df.groupby(institution_column)[column].apply(lambda x: (~x.isnull() & (x != '')).sum())
    
    # Count informative responses (excluding those classified as non-informative)
    informative_responses = df.groupby(institution_column)[column].apply(lambda x: (~x.apply(is_non_informative)).sum())
    
    # Combine the counts into a DataFrame
    response_counts = pd.DataFrame({
        'Total Potential Responses': total_potential_responses,
        'Non-empty Responses': non_empty_responses,
        'Informative Responses': informative_responses
    }).reset_index()
    
    return response_counts

def perform_institution_based_analysis(df, question_mapping, institution_column):
    all_results = []
    
    column_definitions = """
    Column Definitions:
    - Total Potential Responses: All survey participants, including those who left the question blank.
    - Total Responses: All non-empty responses, including both informative and non-informative.
    - Non-empty Responses: Same as Total Responses.
    - Informative Responses: Responses that are not classified as non-informative (e.g., not "N/A", "No comment", etc.).
    """
    
    print(column_definitions)
    
    for question, col in question_mapping.items():
        print(f"\nAnalyzing question: {question}")
        
        response_counts = count_responses_per_institution(df, col, institution_column)
        response_counts['Question'] = question
        
        print("Response counts per institution:")
        print(response_counts.to_string(index=False))
        
        all_results.append(response_counts)
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Reorder columns to put 'Question' first
    columns_order = ['Question', institution_column, 'Total Potential Responses', 'Non-empty Responses', 'Informative Responses']
    combined_results = combined_results[columns_order]
    
    # Save to CSV with column definitions as a comment
    filename = "all_questions_response_counts.csv"
    with open(filename, 'w') as f:
        f.write(f"# {column_definitions.replace(chr(10), chr(10)+'# ')}\n")
        combined_results.to_csv(f, index=False)
    print(f"\nAll results saved to '{filename}'")
    
    return combined_results

# Example usage
if __name__ == "__main__":
    csv_file = "statewide_facultystaff_24.csv"
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3",
        "Please select the reasons for not visiting the campus food pantry.": "Foodpantry_reasons",
        "What are your thoughts about food availability on your campus?": "Foodavail",
        "Please share why you feel unsafe?": "Unsafe_why",
        "Please explain why it is difficult to find housing either on-campus or off-campus?": "Housingdiff_why"
    }

    institution_column = "Institution"

    for col in question_mapping.values():
        df[col] = df[col].apply(preprocess_text)

    results = perform_institution_based_analysis(df, question_mapping, institution_column)
    print("\nCombined results:")
    print(results.to_string(index=False))