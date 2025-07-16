import streamlit as st
import pickle

# Load the model and vectorizer
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Malicious URL Detector", page_icon="🔗")
st.title("Malicious URL Detector")
st.write("Paste a URL below to check whether it's safe or malicious.")

# User input
user_url = st.text_input("Enter a URL here:")

# Prediction
if user_url:
    vec = vectorizer.transform([user_url])
    prediction = model.predict(vec)[0]
    
    if prediction == 0:
        st.success("This URL is Safe!")
    else:
        st.error("Warning: This URL is Malicious!")
