import warnings
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Suppress specific warnings or all warnings
warnings.filterwarnings("ignore")

DB_FAISS_PATH = "vectorstore/db_faiss"

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

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file, encoding="utf-8", delimiter=',')

# Convert the DataFrame rows to a list of Document objects
documents = []
for _, row in df.iterrows():
    content = "\n".join([f"{key}: {value}" for key, value in row.items()])
    documents.append(Document(page_content=content, metadata={"source": csv_file, "row": row["ID"]}))

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
text_chunks = text_splitter.split_documents(documents)

chunks = len(text_chunks)
print(f"Looking through {chunks} text chunks...")

# Download sentence transformers embedding from Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Convert the text chunks into embeddings and save the embeddings into FAISS knowledge base
docusearch = FAISS.from_documents(text_chunks, embeddings)

docusearch.save_local(DB_FAISS_PATH)

def query_system(query, target_id=None):
    # Perform general similarity search
    docs = docusearch.similarity_search(query, k=10)
    output = ""

    if target_id:
        # Filter the results to find the exact match for the ID
        exact_match = next((doc for doc in docs if f"ID: {target_id}" in doc.page_content), None)
        if exact_match:
            output = f"Exact match found in similarity search: {exact_match.page_content}"
        else:
            # Perform a direct lookup in the DataFrame if exact match not found
            exact_match_df = df[df['ID'] == target_id]
            if not exact_match_df.empty:
                exact_match_content = "\n".join([f"{key}: {value}" for key, value in exact_match_df.iloc[0].items()])
                exact_match = Document(page_content=exact_match_content, metadata={"source": csv_file, "row": target_id})
                output = f"Exact match found using direct lookup: {exact_match.page_content}"
            else:
                output = f"Exact match not found. Here are the top results:\n" + "\n".join([doc.page_content for doc in docs])
    else:
        # Parse the query to find relevant columns
        relevant_cols = [col for col in df.columns if "SNAP" in col or "snap" in col]
        if not relevant_cols:
            output = "No relevant columns found for the query."
        else:
            # Filter the DataFrame based on relevant columns
            filtered_df = df[df[relevant_cols].astype(str).apply(lambda x: x.str.contains("1.0|True|Yes", case=False)).any(axis=1)]
            
            if 'how many' in query.lower():
                output = f"{filtered_df.shape[0]} staff members participate in the SNAP program."
            else:
                # Format the results
                results = filtered_df[["ID"] + relevant_cols]
                output = results.to_string(index=False)

    # Append the output to a file
    with open("query_results.txt", "a") as file:
        file.write(f"{query}:\n{output}\n\n")

    return "Results have been written to 'query_results.txt'."

# Example query for general similarity search
query_general = "Which staff members participate in the SNAP program?"
results_general = query_system(query_general)
print(results_general)

# Example query for exact match search
query_exact = "Find the row with ID 11863"
target_id_exact = 11863
results_exact = query_system(query_exact, target_id=target_id_exact)
print(results_exact)

# Example query for count
query_count = "How many staff members participate in the SNAP program?"
results_count = query_system(query_count)
print(results_count)