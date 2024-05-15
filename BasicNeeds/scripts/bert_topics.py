import pandas as pd
import re
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# Load the sentiment analysis results
file_path = "reports/sentiment_analysis_results.txt"
with open(file_path, 'r') as file:
    data = file.read()

# Extract the text responses
pattern = re.compile(r"ID: (\d+)\nText: (.*?)\nSentiment: (.*?)\nScore: (.*?)\n", re.DOTALL)
matches = pattern.findall(data)

# Create a DataFrame
df = pd.DataFrame(matches, columns=["ID", "Text", "Sentiment", "Score"])

# Filter out rows with 'nan' or non-informative texts
df = df[(df['Text'].str.strip() != 'nan') & (df['Text'].str.strip() != '')]

# Load a transformer model for embeddings
model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # or any other suitable model
embedding_model = SentenceTransformer(model_name)

# Perform topic modeling using BERTopic with Hugging Face embeddings
topic_model = BERTopic(embedding_model=embedding_model)
topics, probs = topic_model.fit_transform(df['Text'])

# Add topic information to the DataFrame
df['Topic'] = topics

# Display the topics and their respective counts
topic_info = topic_model.get_topic_info()
print(topic_info)

# Save the topic information to a file
topic_info.to_csv("bertopic_results.csv", index=False)
print("BERTopic results have been saved to 'bertopic_results.csv'")

# If you want to visualize the topics
# Uncomment the following lines if running in an environment that supports visualization
# topic_model.visualize_topics()
# topic_model.visualize_distribution(probs[0])
