import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the response library CSV file
response_library = pd.read_csv('response_library.csv')

# Set up TF-IDF Vectorizer for retrieval
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(response_library['Scenario'])

# Define a function to retrieve responses based on query input
def retrieve_responses(query, top_n=3):
    query_vec = vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return response_library.iloc[related_indices][['Scenario', 'Response', 'Source']]

# Streamlit Interface
st.title("Response Library Search Tool")
query = st.text_input("Enter a scenario or keywords:")

if st.button("Search"):
    if query:
        results = retrieve_responses(query)
        for idx, row in results.iterrows():
            st.write(f"**Scenario**: {row['Scenario']}")
            st.write(f"**Response**: {row['Response']}")
            st.write(f"_Source_: {row['Source']}")
            st.write("---")
    else:
        st.write("Please enter a query.")

