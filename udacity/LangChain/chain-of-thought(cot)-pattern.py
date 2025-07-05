from langchain.llms import OpenAI
import os

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")
completion_model_name = "gpt-3.5-turbo-instruct"
temperature = 0.0 #basically randomness, 0.0 means deterministic output, higher values like 0.7 introduce more variability

# Initialize the OpenAI client with the base URL and API key
llm = OpenAI(
    base_url=base_url,
    api_key=api_key,
    model_name=completion_model_name,
    temperature=temperature)


instruction = """
Andy harvests all the tomatoes from 18 plants that have 7 tomatoes each. If he dries half the
tomatoes and turns a third of the remainder into marinara sauce, how many tomatoes are left?
"""

question1 = """
Karen harvests all the pears from 20 trees that have 10 pears each. She  throws a third of them away as they are rotten,
and turns a quarter of the remaining ones into jam. How many are left?
"""
answer1 = """
    First, let's calculate how many pears Gloria harvests: it's 20 * 10 = 200. 
    Then, let's calculate how many are rotten: 200 * 1/3 = 66.
    Thus, we know how many are left after she throws a third of them away: 200 - 66 = 134.
    1/4 of the remaining ones are turned into jam, or 134 * 1/4 = 33. Therefore, Karen is left with 134 - 33, or 101 pears
"""
question2 = """
Sergei harvests all the strawberries from 50 plants that have 8 stawberries each. He freezes a quarter of them,
and turns half of the remaining ones into jam. How many are left?
"""
answer2 = """
    First, let's calculate how many strawberries Sergei harvests: it's 50 * 8 = 400. 
    Then, let's calculate how many are frozen: 400 * 1/4 = 100.
    Thus, we know how many are left after he freezes 100 of them: 400 - 100 = 300.
    half of the remaining ones are turned into jam, or 300 * 1/2 = 150. Therefore, Sergei is left with 300 - 150, or 150 pears
"""

from langchain.prompts.few_shot import FewShotPromptTemplate

from langchain.prompts import PromptTemplate
example_prompt = PromptTemplate(input_variables=["question", "answer"],
                                template="{question}\n{answer}")
examples = [
    {
        "question": question1,
        "answer": answer1,
    },
    {
        "question": question2,
        "answer": answer2,
    }
]

cot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Use these questions and answers to give correct response to the problem below: {input}",
    input_variables=["input"]
)

cot_text = cot_prompt.format(input=instruction)
print("=== Chain of Thought Prompt ===")
print(cot_text)

print("=== Chain of Thought Answer ===")
print(llm(cot_text))