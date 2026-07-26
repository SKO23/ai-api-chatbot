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
    max-width: 860px !important;
    margin: auto !important;
}

#main-card {
    background: white;
    border-radius: 24px;
    padding: 22px 28px 22px 28px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

#title {
    text-align: center;
    color: #4c1d95;
    margin-bottom: 6px !important;
}

#subtitle {
    text-align: center;
    color: #475569;
    font-size: 16px;
    margin-bottom: 6px !important;
}

#note {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    margin-bottom: 12px !important;
}

/* Chat output box */
#chatbot-box {
    background: #ffffff !important;
    border: 2px solid #8b5cf6 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.08);
}

/* Input area */
#chat-input textarea {
    background: #ffffff !important;
    border: 2px solid #8b5cf6 !important;
    border-radius: 14px !important;
    box-shadow: 0 0 0 4px rgba(139, 92, 246, 0.10);
    color: #111827 !important;
    font-size: 15px !important;
}

#chat-input textarea::placeholder {
    color: #475569 !important;
    opacity: 1 !important;
    font-size: 15px !important;
}

/* Send button */
#send-button {
    background: #6366f1 !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    box-shadow: 0 6px 14px rgba(99, 102, 241, 0.30);
}

#send-button:hover {
    background: #4f46e5 !important;
}

/* Clear button */
#clear-button {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* Helper text */
#helper-text {
    color: #64748b;
    font-size: 12px;
    text-align: center;
    margin-top: 4px !important;
    margin-bottom: 8px !important;
}

/* Examples section */
#example-box {
    background: #f3f4f6 !important;
    border-radius: 14px !important;
    padding: 10px !important;
    margin-top: 8px !important;
}

#example-box button {
    background: #ffffff !important;
    border: 1.5px solid #c4b5fd !important;
    color: #4c1d95 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
}

#example-box button:hover {
    background: #f5f3ff !important;
    border-color: #8b5cf6 !important;
}
"""

@spaces.GPU
def chat_with_ai(message, history):
    if history is None:
        history = []

    if not message or not message.strip():
        return history, ""

    user_message = message.strip()

    history.append({
        "role": "user",
        "content": user_message
    })

    if not api_key:
        ai_reply = "Error: GEMINI_API_KEY is missing. Please check your .env file or Hugging Face Secrets."
    else:
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=user_message
            )
            ai_reply = response.text

        except Exception:
            ai_reply = (
                "The Gemini API is temporarily busy or the free quota/rate limit may have been reached. "
                "Please try again in a few minutes."
            )

    history.append({
        "role": "assistant",
        "content": ai_reply
    })

    return history, ""


def clear_chat():
    return [], ""


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

        chatbot = gr.Chatbot(
            label="Chatbot Conversation",
            height=250,
            elem_id="chatbot-box"
        )

        message_box = gr.Textbox(
            label="Enter your question",
            placeholder="Type your question here... e.g. Explain AI in simple words",
            lines=2,
            elem_id="chat-input"
        )

        with gr.Row():
            send_button = gr.Button("Send Question 🚀", variant="primary", elem_id="send-button")
            clear_button = gr.Button("Clear Chat", elem_id="clear-button")

        gr.Markdown(
            "Tip: Try one of the example prompts below or type your own question.",
            elem_id="helper-text"
        )

        with gr.Group(elem_id="example-box"):
            gr.Examples(
                examples=[
                    "Explain AI in 2 simple lines",
                    "What is an API in simple words?",
                    "Give me 3 beginner AI project ideas",
                    "Explain Gemini API like I am new to coding"
                ],
                inputs=message_box
            )

        send_button.click(
            fn=chat_with_ai,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box]
        )

        message_box.submit(
            fn=chat_with_ai,
            inputs=[message_box, chatbot],
            outputs=[chatbot, message_box]
        )

        clear_button.click(
            fn=clear_chat,
            inputs=None,
            outputs=[chatbot, message_box]
        )

demo.launch(
    theme=gr.themes.Soft(),
    css=custom_css
)