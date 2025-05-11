"""
Simple LLM application with chat models and prompt templates 
This application will translate text from English into another language
"""

import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
apikey = os.getenv("OPENAI_API_KEY")

system_template = "Translate the following from English into {language}"
user_input = input("Enter text to translate: ")
language = input("Enter language to translate: ")

model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": language, "text": user_input})
response = model.invoke(prompt)
print(response.content)