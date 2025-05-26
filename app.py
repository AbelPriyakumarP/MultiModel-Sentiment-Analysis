from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForImageClassification, AutoImageProcessor
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import torch
from PIL import Image
import requests
from io import BytesIO

text_sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
image_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key="gsk_clCsvesVdZowsQSTZ4DbWGdyb3FYAEyuT0H9s8qJy5kEcuBXvadpt")

fusion_prompt = PromptTemplate(
    input_variables=["text_sentiment", "image_sentiment"],
    template="Given text sentiment: {text_sentiment} and image sentiment: {image_sentiment}, provide a combined sentiment analysis for a social media post."
)

fusion_chain = LLMChain(llm=llm, prompt=fusion_prompt)

def analyze_text_sentiment(text):
    result = text_sentiment_model(text)
    return result[0]['label'], result[0]['score']

def analyze_image_sentiment(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = image_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    labels = image_model.config.id2label
    return labels[predicted_class_idx], torch.softmax(logits, dim=1)[0][predicted_class_idx].item()


def multimodal_sentiment_agent(text, image_url):
    # Step 1: Analyze text sentiment
    text_label, text_score = analyze_text_sentiment(text)
    print(f"Text Sentiment: {text_label} (Confidence: {text_score:.2f})")

    # Step 2: Analyze image sentiment
    image_label = analyze_image_sentiment(image_url)
    print(f"Image Sentiment: {image_label}")

    # Step 3: Combine sentiments using LangChain with ChatGroq
    combined_sentiment = fusion_chain.run(
        text_sentiment=f"{text_label} (confidence: {text_score:.2f})", 
        image_sentiment=image_label
        

    )
    return combined_sentiment

