from llama_index.core import SimpleDirectoryReader
import spacy
import pandas as pd
import json
from groq import Groq
import re
import streamlit as st

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load CSV data
df = pd.read_csv("/Users/ashishsharma/Desktop/dreams_interpretations.csv")
dream_data = df.to_dict(orient="records")

# Initialize Groq API
groq_api_key = "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"  # Make sure this is secure in production
llm = Groq(api_key=groq_api_key)

# Keyword extractor
def extract_keywords(text):
    doc = nlp(text)
    keywords = set()
    for chunk in doc.noun_chunks:
        keywords.add(chunk.text.lower())
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN", "VERB"):
            keywords.add(token.lemma_.lower())
    return list(keywords)

# Dream interpreter
def interpret_dream(user_input):
    keywords = extract_keywords(user_input)
    keyword_set = set(keywords)
    matched_entry = None

    for entry in dream_data:
        symbol = str(entry["Dream Symbol"]).lower()
        if symbol in keyword_set:
            matched_entry = entry
            break

    if matched_entry:
        return matched_entry["Interpretation"]
    else:
        system_instruction = (
            "You are a dream assistant. Always respond directly to the user in plain, active voice. "
            "Do not include internal thoughts, reasoning, or any <think>...</think> tags. "
            "If the dream is not found in the database, politely say so and suggest common dream topics like flying, falling, or losing teeth. "
            "Keep responses short, friendly, and clear."
        )

        chat_completion = llm.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            temperature=0.7,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_input},
            ],
        )

        response = chat_completion.choices[0].message.content
        cleaned_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return cleaned_response

# Streamlit UI
st.title("🌙 Sleep Talk")
st.write("Describe your dream and I’ll try to interpret it.")

user_input = st.text_input("What did you dream about?")

if user_input:
    response = interpret_dream(user_input)
    st.markdown("**💭 Interpretation:**")
    st.write(response)
