import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import plotly.express as px
import plotly.graph_objects as go
import string

# Set random seed for reproducibility
np.random.seed(42)

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return text
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to perform topic modeling on open-ended questions
def analyze_topics(df, oe_columns, model_name='all-mpnet-base-v2'):
    # Load the Sentence Transformer model
    embedding_model = SentenceTransformer(model_name)
    
    # Combine all open-ended responses into a single list
    all_texts = df[oe_columns].fillna('').values.flatten()
    all_texts = [text for text in all_texts if text and text not in ["nan", "no", "na", "n/a", "doesn't apply", "not applicable"]]
    
    # Perform topic modeling using BERTopic
    topic_model = BERTopic(embedding_model=embedding_model)
    topics, probs = topic_model.fit_transform(all_texts)
    
    return topic_model, topics, all_texts

# Function to summarize topics
def summarize_topics(topic_model, topics, all_texts):
    topic_info = topic_model.get_topic_info()
    topic_summary = []
    
    for topic in topic_info['Topic']:
        if topic == -1:  # Skip the outlier topic
            continue
        topic_count = sum([1 for t in topics if t == topic])
        topic_texts = [all_texts[i] for i, t in enumerate(topics) if t == topic]
        topic_ids = [i for i, t in enumerate(topics) if t == topic]
        examples = topic_texts[:5]  # Get up to 5 example texts
        
        topic_summary.append({
            'Topic': topic,
            'Count': topic_count,
            'IDs': topic_ids,
            'Examples': examples
        })
    
    return topic_info, topic_summary

# Function to save topic analysis results
def save_topic_results(topic_info, topic_summary, filename="topic_analysis_results.txt"):
    with open(filename, "w") as file:
        file.write("Topic Analysis Results:\n")
        file.write(topic_info.to_string(index=False))
        file.write("\n\nDetailed Topic Summary:\n")
        for topic in topic_summary:
            file.write(f"Topic {topic['Topic']} (Count: {topic['Count']}):\n")
            file.write(f"IDs: {topic['IDs']}\n")
            file.write("Examples:\n")
            for example in topic['Examples']:
                file.write(f"- {example}\n")
            file.write("\n")

# Function to create interactive visualizations
def create_interactive_visualization(topic_info, topic_summary):
    fig = go.Figure()

    for topic in topic_summary:
        fig.add_trace(go.Bar(
            x=[f"Topic {topic['Topic']}"],
            y=[topic['Count']],
            name=f"Topic {topic['Topic']}",
            text=[f"Count: {topic['Count']}<br>Examples:<br>" + "<br>".join(topic['Examples'])],
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Topic Analysis Results",
        xaxis_title="Topics",
        yaxis_title="Count of Responses",
        hovermode="closest"
    )

    fig.show()

# Example usage
if __name__ == "__main__":
    # Example to demonstrate usage; replace with actual file path and columns
    csv_file = "data/statewide_facultystaff_24.csv"
    
    # Load and preprocess the dataset
    df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')
    oe_columns = ["OE1", "OE2", "OE3"]
    question_mapping = {
        "How is food or housing insecurity affecting your work?": "OE1",
        "What could your college or university do to address food and housing insecurity? Please share a solution(s).": "OE2",
        "Is there anything else you would like to share?": "OE3"
    }
    for col in oe_columns:
        df[col] = df[col].apply(preprocess_text)
    
    # Perform topic analysis
    topic_model, topics, all_texts = analyze_topics(df, oe_columns)
    
    # Summarize topics
    topic_info, topic_summary = summarize_topics(topic_model, topics, all_texts)
    
    # Save the topic analysis results
    save_topic_results(topic_info, topic_summary)
    
    # Create interactive visualization
    create_interactive_visualization(topic_info, topic_summary)
    
    print("Topic analysis results have been written to 'topic_analysis_results.txt'.")
    print("Interactive visualizations have been generated.")
