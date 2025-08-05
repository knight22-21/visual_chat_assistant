# backend/langchain_chat.py

import os
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the chat model
chat_model = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192",  # Updated to supported model name
)

# Memory for conversation
memory = ConversationBufferMemory(return_messages=True)

# Conversation chain
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=False
)

def chat_with_summary(summary_text, user_input):
    """
    Use video summary and chat history to respond to the user.
    The summary is added as system prompt only once at the start of the conversation.
    """
    system_prompt = (
        "You are a helpful assistant that helps users understand videos. "
        f"Here is the video summary:\n{summary_text}\n"
    )

    # Add system message once at the beginning
    if not memory.buffer:
        memory.chat_memory.messages.append(SystemMessage(content=system_prompt))

    # Get response from the model
    response = conversation.predict(input=user_input)
    return response
