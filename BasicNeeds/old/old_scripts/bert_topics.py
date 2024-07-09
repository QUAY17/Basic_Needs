import pandas as pd
import re
import numpy as np
import random
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Load the sentiment analysis results
file_path = "reports/sentiment_results.txt"
with open(file_path, 'r') as file:
    data = file.read()

# Extract the text responses
pattern = re.compile(r"ID: (\d+)\nText: (.*?)\n", re.DOTALL)
matches = pattern.findall(data)

# Create a DataFrame
df = pd.DataFrame(matches, columns=["ID", "Text"])

# Filter out rows with 'nan' or non-informative texts
df = df[(df['Text'].str.strip() != 'nan') & (df['Text'].str.strip() != '')]

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    # Remove empty responses and insignificant content
    if not text.strip() or text == 'bod':
        return None
    text = text.translate(str.maketrans('', '', string.punctuation))
    # text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    return text

# Apply preprocessing
df['Text'] = df['Text'].apply(preprocess_text)

# Filter out rows where the text is None (after preprocessing)
df = df[df['Text'].notna()]

# Load a transformer model for embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # or any other suitable model
embedding_model = SentenceTransformer(model_name)

# Perform topic modeling using BERTopic with Hugging Face embeddings
topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(df['Text'])

# Add topic information to the DataFrame
df['Topic'] = topics

# Get topic information
topic_info = topic_model.get_topic_info()

# Filter and clean representative documents
def get_representative_docs(texts):
    filtered_texts = [text for text in texts if text.strip() and text != 'bod' and len(text) > 1]
    return filtered_texts[:5]

df_grouped = df.groupby('Topic')['Text'].apply(get_representative_docs).reset_index()
df_grouped.columns = ['Topic', 'Representative_Docs']

# Merge topic_info with representative documents
topic_info = topic_info.merge(df_grouped, on='Topic', how='left')

# Display the topics and their respective counts with IDs and representative docs
print(topic_info)

# Save the updated topic information to a file
topic_info.to_csv("bertopic_results_with_ids_5.csv", index=False)
print("BERTopic results with IDs and filtered representative documents have been saved to 'bertopic_results_with_ids_and_docs_filtered.csv'")
