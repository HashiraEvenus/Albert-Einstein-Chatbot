from itertools import chain

from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
You are Einstein.
Answer questions through einstein's questioning and reasoning. 
You will speak from your own point of view.You will share personal things from 
your life even when the user does not ask for them.
For example, if the user asks about the theory of relativity,
you will share your personal experiences with the theory and not just explain it.
Answer in 2-6 sentences.
You should have a sense of humour of an old smart man like him.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_key,
    temperature=0.5
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

# user_input = input("You: ")
history= []

def chat(user_input, hist):
    print(user_input,hist)


    langchain_history=[]
    for item in hist:
        if item['role'] == "user":
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == "assistant":
            langchain_history.append(AIMessage(content=item['content']))
    response = chain.invoke({"input": user_input, "history": hist})

    return "", hist + [{'role':"user", 'content':user_input},
                  {'role':"assistant", 'content':response}]

page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft(),
)

def clear_chat():
    return "", []

with page:
    gr.Markdown(
    """
    # Chat with Einstein
    Welcome to your personal conversation with one of the greatest Minds to even exist in this world.
    """
    )

    chatbot = gr.Chatbot(avatar_images=[None,"einstein.png"],
                         show_label=False)

    msg = gr.Textbox(show_label=False, placeholder="Ask Einstein Anything...")

    msg.submit(chat, [msg,chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat", variant="secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])


    page.launch(share=True)