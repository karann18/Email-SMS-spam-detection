import streamlit as st
import pickle
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


try:
    nltk.data.find('tokenizers/punkt')          # Check if punkt tokenizer is available for now we will use punkt
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')         # Check if stopwords are available
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()                            # reducing words to their root form

# Function to preprocess text
def transform_text(text):
    text = text.lower()                         # Convert to lowercase
    text = nltk.word_tokenize(text)
    y = []

    # Remove non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI(simple ui with text space and predict button)
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

predict_button = st.button("Predict")

if input_sms:
    # Preprocess
    transformed_sms = transform_text(input_sms)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)[0]

    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

