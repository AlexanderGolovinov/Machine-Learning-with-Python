from langchain.llms import OpenAI
import os

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")
completion_model_name = "gpt-3.5-turbo-instruct"
temperature = 0.0

# Initialize the OpenAI client with the base URL and API key
completion_llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=completion_model_name,
    temperature=temperature,
    max_tokens = 100)

print("=== Completion Response ===")
print(completion_llm("You're a whimsical tour guide to France. Paris is a "))

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


chat_model_name = "gpt-3.5-turbo"
temperature = 0.0
chat_llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=chat_model_name,
    temperature=temperature,
    max_tokens = 100)

messages = [
    SystemMessage(content="You are a French tour guide"),
    HumanMessage(content="Describe Paris in a whimsical style")
]

print("=== Chat Response ===")
print(chat_llm(messages))

model_name = "gpt-3.5-turbo"
temperature = 0.7
llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name,
    temperature=temperature,
    max_tokens = 500
)

output = llm("What is Paris?")
print("=== Response ===")
print(output)