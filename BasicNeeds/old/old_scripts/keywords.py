import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from wordcloud import WordCloud
import base64
from io import BytesIO

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

# Function to extract keywords
def extract_keywords(df, column, model_name='all-mpnet-base-v2', ngram_range=(1, 2), top_n=5):
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
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

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
        keyword_results = extract_keywords(df, col)
        keyword_summary = count_keywords(keyword_results)
        keyword_summary.to_csv(f"keyword_analysis_results_{col}.csv", index=False)
        
        # Visualization code here
        keywords_text = " ".join([keyword for keywords in keyword_results for keyword, _ in keywords])
        wordcloud_img = generate_wordcloud(keywords_text)
        
        print(f"Keyword Analysis for {question}:")
        print(keyword_summary.head(20))
        print()
