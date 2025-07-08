from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.chains.question_answering import load_qa_chain

import os

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")
model_name = "gpt-3.5-turbo-instruct"

loader = CSVLoader(file_path='../../data/tv-reviews.csv')
docs = loader.load()

llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name)

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

db = Chroma.from_documents(split_docs, embeddings)
query = """
    Based on the reviews in the context, tell me what people liked about the picture quality.
    Make sure you do not paraphrase the reviews, and only use the information provided in the reviews.
    """

use_chain_helper = False
if use_chain_helper:
    rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
    print(rag.run(query))
else:
    similar_docs = db.similarity_search(query, k=5)
    prompt = PromptTemplate(
        template="{query}\nContext: {context}",
        input_variables=["query", "context"],
    )
    chain = load_qa_chain(llm, prompt = prompt, chain_type="stuff")
    print(chain.run(input_documents=similar_docs, query = query))