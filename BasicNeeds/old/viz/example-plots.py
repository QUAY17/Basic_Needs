"""import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample data
df = pd.read_csv('data/statewide_facultystaff_24.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a sample figure
fig = px.histogram(df, x='Age', title='Age Distribution')

# Layout of the app
app.layout = html.Div(children=[
    html.H1(children='Survey Data Dashboard'),

    dcc.Graph(
        id='age-distribution',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
"""


import streamlit as st
import pandas as pd
import plotly.express as px

# Load data
df = pd.read_csv('data/statewide_facultystaff_24.csv')

# Title
st.title('Survey Data Dashboard')

# Histogram
fig = px.histogram(df, x='Age', title='Age Distribution')
st.plotly_chart(fig)

# Show data
st.write(df)

"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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

# List of binary columns
binary_cols = [
    'USDA1', 'USDA2', 'USDA3', 'USDA4', 'USDA5', 'USDA6', 'USDA7', 'USDA8', 'USDA9', 'USDA10',
    'USDA11', 'USDA12', 'USDA13', 'USDA14', 'USDA15', 'USDA16', 'USDA17', 'USDA18', 'FIBin', 'SNAP', 
    'WIC', 'TANF', 'FDPIR', 'NSLP', 'SBP', 'SFSP', 'Foodbank', 'Other', 'None', 'Foodpantry', 
    'Nutrmeals', 'HIBin', 'Homeless_SR', 'Trans_public', 'GADBin', 'PHQBin', 'Socialsupp', 'Healthserv', 
    'Insurance', 'Nutrsec', 'Nutrbin'
]

# Convert binary columns to numeric
for col in binary_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1'] else 0)

# List of categorical columns
categorical_cols = [
    'UserLanguage', 'Age', 'Institution', 'Inst_Type', 'Type', 'TypeA', 'Type_other', 'Type2', 'Type3',
    'Children', 'FICat', 'Foodprog', 'Mealplan', 'Cost', 'Safety', 'Homeless_calc', 'Oncampus', 
    'Transportation', 'GAD1', 'GAD2', 'PHQ1', 'PHQ2', 'Levelstudy', 'Levelstudy_other', 'Fieldstudy', 
    'Fieldstudy_other', 'Race', 'Race_other', 'Race2', 'Native', 'Native_other', 'Residency', 
    'Moreexpensive', 'Nmresident', 'Gender', 'Gender_other', 'Gender2', 'Transgender', 'Sexuality', 
    'Sexuality_other', 'Sexuality2', 'Military', 'Income', 'Income_other'
]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# List of open-ended columns for one-hot encoding
open_ended_cols = [
    'Obstacles', 'Foodpantry_reasons', 'Foodpantry_reasons_other', 'Foodavail', 'Placesslept', 
    'Placesslept_other', 'Housingdiff_why', 'Strategies', 'Strategies_other', 'Experience', 
    'Trans_pub_notuse', 'Trans_pub_notuse_why', 'Unsafe_why', 'OE1', 'OE2', 'OE3'
]

# One-hot encode open-ended columns
encoder = OneHotEncoder(sparse_output=False)
encoded_cols = encoder.fit_transform(df[open_ended_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(open_ended_cols))

# Replace non-numeric values with NaN
df.replace({'.': np.nan}, inplace=True)

# Fill missing values with 0 (assuming that NaN means no response or not applicable)
df.fillna(0, inplace=True)

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Generate correlation matrix
corr = df.corr()

# Plot heatmap of the correlation matrix

plt.figure(figsize=(20, 20))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()
"""