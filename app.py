import os
import gradio as gr
import spaces
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

@spaces.GPU
def chat_with_ai(message, history):
    if not message or not message.strip():
        return "Please type a question."

    if not api_key:
        return "Error: GEMINI_API_KEY is missing. Please add it in Hugging Face Secrets."

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=message
        )
        return response.text

    except Exception as e:
        return f"Gemini API error: {str(e)}"

demo = gr.ChatInterface(
    fn=chat_with_ai,
    title="Gemini API Chatbot",
    description="A simple chatbot built with Python, Gradio and the Google Gemini API."
)

demo.launch()