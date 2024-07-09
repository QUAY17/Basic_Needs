import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.preprocessing import OneHotEncoder
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress specific FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Function to remove BOM from a CSV file
def remove_bom(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(file_path, 'wb') as file:
        file.write(content)

# Remove BOM from the CSV file if present
csv_file = "data/statewide_facultystaff_24.csv"
remove_bom(csv_file)

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

# Sample 10% of the data for testing
logging.info("Sampling data...")
df_sampled = df.sample(frac=0.1, random_state=42)

# Identify open-ended columns
open_ended_cols = ['OE1', 'OE2', 'OE3', 'Type_other', 'Foodpantry_reasons_other', 'Placesslept_other', 'Strategies_other', 'Experience', 'Trans_pub_notuse_why', 'Disability_other']

# Create binary columns indicating whether each open-ended column contains data
logging.info("Creating binary columns for open-ended questions...")
for col in open_ended_cols:
    df_sampled[col + '_answered'] = df_sampled[col].notna().astype(int)

# Drop original open-ended columns
df_sampled = df_sampled.drop(columns=open_ended_cols)

# Convert categorical data to binary/one-hot encoded format
logging.info("Encoding categorical data...")
categorical_cols = df_sampled.select_dtypes(include=['object']).columns
df_categorical = df_sampled[categorical_cols].fillna('Missing')
encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(df_categorical)

# Combine encoded categorical data with original DataFrame (excluding original categorical columns)
df_encoded = pd.concat([df_sampled.drop(columns=categorical_cols).reset_index(drop=True), 
                        pd.DataFrame(encoded_categorical, 
                        columns=encoder.get_feature_names_out(categorical_cols)).reset_index(drop=True)], axis=1)

# Convert binary columns to boolean
logging.info("Converting binary columns to boolean...")
binary_cols = df_encoded.select_dtypes(include=['int64', 'float64']).columns
df_binary = df_encoded[binary_cols].astype(bool)

# Log the shape of the final DataFrame
logging.info(f"Shape of encoded DataFrame: {df_binary.shape}")

# Convert DataFrame to a dense matrix for testing
df_dense = df_binary.astype(int)

# Apply the Apriori algorithm
logging.info("Applying Apriori algorithm...")
frequent_itemsets = apriori(df_dense, min_support=0.1, use_colnames=True)

# Extract association rules
logging.info("Extracting association rules...")
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Save the association rules to a file
logging.info("Saving association rules to file...")
rules.to_csv("association_rules.csv", index=False)

logging.info("Association rules have been saved to 'association_rules.csv'.")
