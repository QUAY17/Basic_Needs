import pandas as pd
import re
from transformers import BertTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Function to parse the sentiment analysis results text file
def parse_sentiment_results(file_path):
    results = {"OE1": [], "OE2": [], "OE3": []}
    current_section = None

    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "Sentiment Analysis for" in line:
                current_section = line.strip().split()[-1][:-1]
            elif line.startswith("ID:"):
                id_ = int(line.split(":")[1].strip())
            elif line.startswith("Text:"):
                text = line.split(":")[1].strip()
            elif line.startswith("Sentiment:"):
                sentiment = line.split(":")[1].strip()
            elif line.startswith("Score:"):
                score = float(line.split(":")[1].strip())
                if current_section in results:
                    results[current_section].append((id_, text, sentiment, score))

    return results

# Parse the text file
file_path = "reports/sentiment_analysis_results.txt"
sentiment_results = parse_sentiment_results(file_path)

# Convert results to DataFrame
data = []
for col, results in sentiment_results.items():
    for id_, text, sentiment, score in results:
        data.append({"ID": id_, "Question": col, "Text": text, "Sentiment": sentiment, "Score": score})

df_sentiment = pd.DataFrame(data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = tokenizer.tokenize(text)
    return ' '.join(tokens)

# Apply preprocessing
df_sentiment['Cleaned_Text'] = df_sentiment['Text'].apply(lambda x: preprocess_text(str(x)) if x != 'nan' else '')

# Filter out empty texts
df_sentiment = df_sentiment[df_sentiment['Cleaned_Text'] != '']

# Combine all texts for each question
combined_texts = {
    "OE1": ' '.join(df_sentiment[df_sentiment['Question'] == "OE1"]['Cleaned_Text']),
    "OE2": ' '.join(df_sentiment[df_sentiment['Question'] == "OE2"]['Cleaned_Text']),
    "OE3": ' '.join(df_sentiment[df_sentiment['Question'] == "OE3"]['Cleaned_Text'])
}

# Function to extract keywords
def extract_keywords(text, n=10):
    vectorizer = TfidfVectorizer(max_features=n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    keywords = sorted(list(zip(feature_names, scores)), key=lambda x: x[1], reverse=True)
    return keywords

# Extract keywords for each question
for question, text in combined_texts.items():
    keywords = extract_keywords(text, n=10)
    print(f"\nKeywords for {question}:")
    for word, score in keywords:
        print(f"{word}: {score:.4f}")

# Function to perform LDA
def perform_lda(text, n_topics=5, n_words=10):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    topics = []
    for idx, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_words:]]
        topics.append(f"Topic {idx + 1}: {' '.join(topic_words)}")
    return topics

# Perform LDA for each question
for question, text in combined_texts.items():
    topics = perform_lda(text, n_topics=5, n_words=10)
    print(f"\nTopics for {question}:")
    for topic in topics:
        print(topic)
