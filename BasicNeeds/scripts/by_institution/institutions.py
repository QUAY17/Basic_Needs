import pandas as pd
import numpy as np
from collections import Counter
import os

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
        "unsure", "not sure", "nothing", "none", "n/a", "N/A", "i don't know", "prefer not to say", "no comment",
        "im not sure", "dont know", "i am not sure", "i'm not too sure", "i am not knowledgeable about this", "i have no idea",
        "sorry no ideas", "i have no comment", "i am unsure", "i have no suggestions", "not applicable", "not at this time", "no thank you", "na"
    ]
    return any(response in text.lower() for response in non_informative_responses)

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

# Function to count non-empty responses for each question, excluding null/NaN and empty strings
def count_non_empty_responses_per_question(df, question_columns):
    response_counts = {}
    
    for column in question_columns:
        # Count non-empty responses (excluding null/NaN and empty strings)
        non_empty_responses = (~df[column].isnull() & (df[column] != '')).sum()
        response_counts[column] = non_empty_responses  # Store non-empty counts
    
    return response_counts

def list_responses_per_institution(df, column, institution_column):
    # Filter out non-informative responses
    informative_df = df[~df[column].apply(is_non_informative)]
    
    # Group responses by institution
    grouped_responses = informative_df.groupby(institution_column)[column].apply(list).reset_index()
    grouped_responses.columns = [institution_column, 'Responses']
    
    return grouped_responses

def perform_institution_based_analysis(df, question_mapping, institution_column):
    all_results = []
    
    column_definitions = """
    Column Definitions:
    - Total Potential Responses: All survey participants, including those who left the question blank.
    - Non-empty Responses: All responses that are not null or empty string.
    - Informative Responses: Responses that are not classified as non-informative (e.g., not "N/A", "No comment", etc.).
    - Response: Individual informative responses, with each response on a separate row.
    """
    
    #print(column_definitions)
    
    for question, col in question_mapping.items():
        print(f"\nAnalyzing question: {question}")
        
        response_counts = count_responses_per_institution(df, col, institution_column)
        responses = list_responses_per_institution(df, col, institution_column)
        
        # Merge counts and responses
        combined = pd.merge(response_counts, responses, on=institution_column)
        
        # Create the final dataframe with the desired structure
        final_data = []
        for _, row in combined.iterrows():
            # Add the row with count information
            final_data.append({
                'Question': question,
                'Institution': row[institution_column],
                'Total Potential Responses': row['Total Potential Responses'],
                'Non-empty Responses': row['Non-empty Responses'],
                'Informative Responses': row['Informative Responses'],
                'Response': row['Responses'][0] if row['Responses'] else ''
            })
            # Add rows for additional responses
            for response in row['Responses'][1:]:
                final_data.append({
                    'Question': '',
                    'Institution': '',
                    'Total Potential Responses': '',
                    'Non-empty Responses': '',
                    'Informative Responses': '',
                    'Response': response
                })
        
        question_results = pd.DataFrame(final_data)
        all_results.append(question_results)
    
    # Combine all results into a single DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save to CSV with column definitions as a comment
    filename = "faculty_counts.csv"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# {column_definitions.replace(chr(10), chr(10)+'# ')}\n")
        combined_results.to_csv(f, index=False, quoting=1)  # quoting=1 ensures all fields are quoted
    print(f"\nAll results saved to '{filename}'")
    
    return combined_results

# Example usage
if __name__ == "__main__":
    csv_file = "faculty_data.csv"
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
    # print("\nCombined results (sample):")
    # print(results.head(10).to_string(index=False))