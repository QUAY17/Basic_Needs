import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
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
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not applicable", "not at this time", "no thank you"
    ]
    if any(response in text for response in non_informative_responses):
        return False
    return True

# Function to analyze topics
def analyze_topics(df, column, model_name='all-mpnet-base-v2', min_topic_size=10, ngram_range=(1, 2)):
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

# Example usage
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
    
    for question, col in question_mapping.items():
        topic_model, topics, texts = analyze_topics(df, col)
        topic_info, topic_summary = summarize_topics(topic_model, topics, texts)
        topic_info.to_csv(f"topic_analysis_results_{col}.csv", index=False)
        # Visualization code here
        print(f"Topic Analysis for {question}:")
        for summary in topic_summary:
            print(f"Topic {summary['Topic']} (Count: {summary['Count']})")
            for example in summary['Examples']:
                print(f"- {example}")
            print()
