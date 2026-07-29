import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk


nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

ps = PorterStemmer()

# -------------------------------
# Text Preprocessing Function
# -------------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    # Remove special characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Spam SMS Detector", page_icon="📩")

st.title("📩 Spam SMS Detection")
st.write("Enter an SMS below to check whether it is **Spam** or **Not Spam**.")

input_sms = st.text_area("Enter SMS")

if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        transformed_sms = transform_text(input_sms)

        vector_input = tfidf.transform([transformed_sms])

        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
