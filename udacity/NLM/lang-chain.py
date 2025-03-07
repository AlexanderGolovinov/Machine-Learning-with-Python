from langchain_openai import ChatOpenAI
import os

# llm = ChatOpenAI()
# llm.invoke("Hello, world!")

kApiKey = os.getenv("TEST_NVCF_API_KEY", "")
print(kApiKey)