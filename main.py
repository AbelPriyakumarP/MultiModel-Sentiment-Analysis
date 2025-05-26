import streamlit as st
from app import analyze_text_sentiment, analyze_image_sentiment, multimodal_sentiment_agent
import requests
from PIL import Image
from io import BytesIO

# Streamlit app configuration
st.set_page_config(page_title="Multimodal Sentiment Analysis", layout="wide")

# Title and description
st.title("Multimodal Sentiment Analysis for Social Media")
st.write("Enter a text (e.g., a tweet or post) and an image URL to analyze the combined sentiment.")

# Input fields
text_input = st.text_area("Enter your text:", placeholder="e.g., I love the new design of this product!")
image_url = st.text_input("Enter image URL:", placeholder="e.g., https://example.com/happy_image.jpg")
submit_button = st.button("Analyze Sentiment")

# Process inputs and display results
if submit_button:
    if not text_input or not image_url:
        st.error("Please provide both text and an image URL.")
    else:
        try:
            # Display input image
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="Input Image", use_column_width=True)

            # Analyze sentiments
            with st.spinner("Analyzing sentiments..."):
                text_label, text_score = analyze_text_sentiment(text_input)
                image_label = analyze_image_sentiment(image_url)
                combined_sentiment = multimodal_sentiment_agent(text_input, image_url)

            # Display results
            st.subheader("Results")
            st.write(f"**Text Sentiment**: {text_label} (Confidence: {text_score:.2f})")
            st.write(f"**Image Sentiment**: {image_label}")
            st.write(f"**Combined Sentiment**: {combined_sentiment}")

        except Exception as e:
            st.error(f"Error processing request: {str(e)}")

# Footer
st.markdown("---")