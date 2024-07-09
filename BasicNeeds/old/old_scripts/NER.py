import pandas as pd
from transformers import BertTokenizer, BertForTokenClassification, pipeline
import warnings

# Suppress specific warnings or all warnings
warnings.filterwarnings("ignore")

# Load the CSV file
def remove_bom(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(file_path, 'wb') as file:
        file.write(content)

csv_file = "data/statewide_facultystaff_24.csv"
remove_bom(csv_file)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

# Load pre-trained model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForTokenClassification.from_pretrained(model_name)

# Initialize the NER Named Entity Recognition pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Define the columns to analyze
text_columns = ["OE1", "OE2", "OE3"]

# Prepare to write results to a file
output_file = "ner_results.txt"
with open(output_file, "w") as file:
    for col in text_columns:
        file.write(f"Named Entities for {col}:\n")
        for idx, text in df[col].dropna().items():
            entities = ner_pipeline(text)
            file.write(f"ID: {df.at[idx, 'ID']}\nText: {text}\nEntities: {entities}\n\n")

print(f"NER results have been written to '{output_file}'")
