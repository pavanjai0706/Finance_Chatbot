import streamlit as st
import pandas as pd
import pdfplumber
import re
from transformers import pipeline


qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def extract_text_from_pdf(pdf_path):
    text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:15]:
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if page_text:
                text.append(page_text)

    extracted_text = "\n\n".join(text)
    
    if not extracted_text.strip():
        return "No text found in PDF. Try uploading a different file."
    
    return extracted_text


def extract_relevant_text(text, keyword):
    lines = text.split("\n")
    relevant_lines = [line for line in lines if keyword.lower() in line.lower()]
    
    if not relevant_lines:
        return " ".join(text.split()[:500])
    
    return "\n".join(relevant_lines[:5])


def load_excel_data(excel_path):
    return pd.read_excel(excel_path)


def get_financial_data(df, query):
    query = query.lower()
    words = query.split()
    best_match = None
    best_score = 0

    for col in df.columns:
        col_lower = col.lower()
        match_score = sum(1 for word in words if word in col_lower)

        if match_score > best_score:
            best_match = col
            best_score = match_score

    year_match = re.search(r"\d{4}", query)
    if best_match and year_match:
        year = int(year_match.group())
        if "year" in df.columns and year in df["Year"].values:
            row = df[df["Year"] == year]
            return f"{best_match} in {year}: {row[best_match].values[0]}"

    return f"Best match: {best_match}, but no exact data found."


st.title("Finance Chatbot for 10-K/10-Q Reports")

uploaded_file = st.file_uploader("Upload a PDF or Excel file below", type=["pdf", "xlsx"])

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]

    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
        
        if "No text found" in text:
            st.error(text)
        else:
            user_question = st.text_input("Ask me a question:")
            if user_question:
                filtered_text = extract_relevant_text(text, user_question)
                
                if not filtered_text.strip():
                    st.error("No relevant information found. Try rephrasing your question.")
                else:
                    answer = qa_pipeline(question=user_question, context=filtered_text)
                    st.subheader("Answer:")
                    st.write(answer["answer"])

    elif file_type == "xlsx":
        df = load_excel_data(uploaded_file)
        st.subheader("Extracted Financial Data:")
        st.write(df.head())

        query = st.text_input("Ask about financial data (e.g., 'Tell me about the company')")
        if st.button("Get Data"):
            st.subheader("Answer:")
            st.write(get_financial_data(df, query))
