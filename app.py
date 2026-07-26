import os
import gradio as gr
import spaces
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

custom_css = """
body {
    background: linear-gradient(135deg, #f0f9ff, #fdf4ff);
}

.gradio-container {
    max-width: 950px !important;
    margin: auto !important;
}

#main-card {
    background: white;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

#title {
    text-align: center;
    color: #6d28d9;
}

#subtitle {
    text-align: center;
    color: #475569;
    font-size: 18px;
}

#note {
    text-align: center;
    color: #64748b;
    font-size: 14px;
}

/* Make chatbot area clearer */
#chatbot-box {
    border: 2px solid #ddd6fe !important;
    border-radius: 16px !important;
    background: #faf5ff !important;
}

/* Make input box more visible */
#chat-input textarea {
    background: #ffffff !important;
    border: 2px solid #8b5cf6 !important;
    border-radius: 14px !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.10);
    color: #111827 !important;
    font-size: 16px !important;
}

#chat-input textarea::placeholder {
    color: #475569 !important;
    opacity: 1 !important;
    font-size: 16px !important;
}

/* Make example prompts clearer */
button[aria-label="Use example"] {
    background: #ffffff !important;
    border: 1.5px solid #ddd6fe !important;
    color: #5b21b6 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}

button[aria-label="Use example"]:hover {
    background: #f5f3ff !important;
    border-color: #8b5cf6 !important;
}
"""

@spaces.GPU
def chat_with_ai(message, history):
    if not message or not message.strip():
        return "Please type a question."

    if not api_key:
        return "Error: GEMINI_API_KEY is missing. Please check your .env file or Hugging Face Secrets."

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=message
        )
        return response.text

    except Exception as e:
        return f"Gemini API error: {str(e)}"


with gr.Blocks(title="Gemini API Chatbot") as demo:

    with gr.Column(elem_id="main-card"):
        gr.Markdown("# 💬 Gemini API Chatbot", elem_id="title")

        gr.Markdown(
            "Ask a question and the app will call the Gemini API to generate an answer.",
            elem_id="subtitle"
        )

        gr.Markdown(
            "Built with Python, Gradio, Gemini API, GitHub and Hugging Face Spaces.",
            elem_id="note"
        )

        gr.ChatInterface(
            fn=chat_with_ai,
            chatbot=gr.Chatbot(
                height=420,
                label="Chatbot Conversation",
                elem_id="chatbot-box"
            ),
            textbox=gr.Textbox(
                placeholder="Type your question here... e.g. Explain AI in simple words",
                label="Your question",
                elem_id="chat-input"
            ),
            examples=[
                "Explain AI in 2 simple lines",
                "What is an API in simple words?",
                "Give me 3 beginner AI project ideas",
                "Explain Gemini API like I am new to coding"
            ]
        )

demo.launch(
    theme=gr.themes.Soft(),
    css=custom_css
)