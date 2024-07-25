import streamlit as st
import pickle
from sklearn.pipeline import Pipeline

# Load the model
with open('text_clf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Streamlit app
st.title("Fake News Detection")

st.write("""
## Enter the news article text below to predict if it's real or fake.
""")

# Text input
user_input = st.text_area("Enter news article text here", "")

# Prediction
if st.button("Predict"):
    if user_input.strip() == "":
        st.write("Please enter some text to predict.")
    else:
        # The model pipeline handles vectorization internally
        prediction = model.predict([user_input])
        result = "Fake News" if prediction[0] == 1 else "Real News"
        st.write(f"The news article is: **{result}**")
