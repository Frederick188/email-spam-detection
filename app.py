import streamlit as st
import joblib

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# App title
st.title("📧 Email Spam Detection App")

# User input
user_input = st.text_area("Enter the email or SMS message below:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Transform the input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)

        # Display result
        if prediction[0] == 1:
            st.error("🚫 This is a Spam message!")
        else:
            st.success("✅ This is a Not Spam (Ham) message.")

# Optional: Footer
st.markdown("---")
st.markdown("Made with using Streamlit and Naive Bayes")
