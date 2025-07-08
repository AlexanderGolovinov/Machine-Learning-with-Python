import os

from langchain.chat_models import ChatOpenAI # this is the new import statement
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, NonNegativeInt
from typing import List
from random import sample
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path='../../data/tv-reviews.csv')
data = loader.load()

print(data)

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")
model_name = "gpt-3.5-turbo-instruct"
# llm = OpenAI(model_name=model_name, temperature=0) - This has been replaced with the next line of code
llm = ChatOpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=model_name)


class ReviewSentiment(BaseModel):
    positives: List[NonNegativeInt] = Field(description="index of a positive TV review, starting from 0")
    negatives: List[NonNegativeInt] = Field(description="index of a negative TV review, starting from 0")


parser = PydanticOutputParser(pydantic_object=ReviewSentiment) #pydantic_object is the model we want to parse the output into
# The PydanticOutputParser will parse the output of the LLM into the ReviewSentiment model
print(parser.get_format_instructions())

prompt = PromptTemplate(
    template="{question}\n{format_instructions}\nContext: {context}",
    input_variables=["question", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions},
)
question = """
    Classify TV reviews provided in the context into positive and negative.
    Only use the reviews provided in this context, do not make up new reviews or use any existing information you know about these TVs.
    If there are no positive or negative reviews, output an empty JSON array.
"""

reviews_to_classify = sample(data, 3)
context = '\n'.join(review.page_content for review in reviews_to_classify)

query = prompt.format(context = context, question = question)
print(query)

output = llm.predict(query) #the "predict" has been added

print(output)
result = parser.parse(output)
print(result)
print("Positives:\n" + "\n".join([reviews_to_classify[i].page_content for i in result.positives]))
print("Negatives:\n" + "\n".join([reviews_to_classify[i].page_content for i in result.negatives]))