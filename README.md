# 🚀 Multimodal Sentiment Analysis Agent using GenAI & LangChain

Welcome to a hands-on project exploring the fusion of **Generative AI, Vision Transformers, and LangChain** for **Multimodal Sentiment Analysis**! This application evaluates both **text** and **image sentiments** from a social media post and intelligently combines them using a conversational LLM workflow.

---

## 📊 Project Overview

This project demonstrates how to integrate:

* 📝 **Text Sentiment Analysis**: Using `cardiffnlp/twitter-roberta-base-sentiment-latest` via Hugging Face Transformers.
* 🖼️ **Image Sentiment Classification**: Powered by `google/vit-base-patch16-224` Vision Transformer.
* 🤖 **LLM Fusion & Decision-Making**: Combining text and image sentiment insights with **Llama 3 (70B)** model on **ChatGroq** through **LangChain’s LLMChain** and **PromptTemplate**.

---

## 📺 Demo Video

> 📺 Check out a quick demonstration of the application in action:
> ![[Video alt](https://github.com/AbelPriyakumarP/MultiModel-Sentiment-Analysis/blob/574edf5f6fb5cd6a3b3e7c3f47e79ddfa731caad/mulitmodel%20sentiment-analysis.mp4)]

---

## 📌 Key Features

* 📃 Real-time **text sentiment analysis**
* 🖼️ On-the-fly **image sentiment classification**
* 🔗 **LangChain-based multimodal reasoning**
* 💬 Dynamic fusion of sentiment outputs through LLMChain
* ⚙️ Clean, modular Python implementation

---

## 🛠️ Tech Stack

* **Python**
* **Hugging Face Transformers**
* **Vision Transformers (ViT)**
* **LangChain**
* **ChatGroq**
* **Llama 3 (70B)**

---

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/multimodal-sentiment-agent.git
   cd multimodal-sentiment-agent
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Add your Groq API key inside the `app.py`:

   ```python
   groq_api_key = "your_groq_api_key"
   ```

4. Run the application:

   ```bash
   python app.py
   ```

---

## 📈 How It Works

1. **Analyze text sentiment** using a fine-tuned RoBERTa model.
2. **Classify image sentiment** with a ViT model.
3. **Combine both results** with an LLMChain powered by Llama 3 on Groq.
4. **Generate a unified sentiment assessment** tailored for social media context.

---

## 💡 Use Cases

* Digital marketing sentiment analysis
* Social media monitoring
* Brand reputation tracking
* Content moderation pipelines
* Multimodal GenAI research projects

---

## 🤝 Let’s Connect!

If you're passionate about **Generative AI, LangChain, LLMs, or Multimodal Agents**, feel free to connect:

* 📱 [LinkedIn]([https://www.linkedin.com/in/yourprofile](https://www.linkedin.com/in/abel-priyakumar-p/))

---

## 📌 Hashtags for Reach

\#GenerativeAI #LangChain #LLM #MultimodalAI #SentimentAnalysis #VisionTransformer #Groq #Llama3 #LangChainAgents #LLMOps #MachineLearning #ArtificialIntelligence #DeepLearning #GenAIProjects #AIInnovation #PromptEngineering #OpenSourceAI #PythonProjects

---

## 📜 License

This project is open-sourced under the MIT License.
