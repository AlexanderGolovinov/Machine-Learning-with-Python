import pandas as pd

# In[2]:


df = pd.read_csv('./data/2023_fashion_trends.csv')
print(f"Dataset shape: {df.shape}")
df.head()

import re
def clean_text(text):
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub('<.*?>', ' ', text)
        # Remove non-alphanumeric characters
        text = re.sub('[^a-zA-Z0-9]', ' ', text)
        # Remove extra whitespaces
        text = re.sub('\s+', ' ', text).strip()
        return text
    return ""


# Apply cleaning to the 'Text' column and create our 'text' column
df['text'] = df['Trends'].apply(clean_text)
df = df[df['text'].str.len() > 0]

import openai

openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = ""

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
batch_size = 100
embeddings = []
for i in range(0, len(df), batch_size):
    # Send text data to OpenAI model to get embeddings
    response = openai.Embedding.create(
        input=df.iloc[i:i + batch_size]["text"].tolist(),
        engine=EMBEDDING_MODEL_NAME
    )

    # Add embeddings to list
    embeddings.extend([data["embedding"] for data in response["data"]])

# Add embeddings list to dataframe
df["embeddings"] = embeddings
df


from openai.embeddings_utils import get_embedding, distances_from_embeddings


def get_rows_sorted_by_relevance(question, df):
    """
    Function that takes in a question string and a dataframe containing
    rows of text and associated embeddings, and returns that dataframe
    sorted from least to most relevant for that question
    """

    # Get embeddings for the question text
    question_embeddings = get_embedding(question, engine=EMBEDDING_MODEL_NAME)

    # Make a copy of the dataframe and add a "distances" column containing
    # the cosine distances between each row's embeddings and the
    # embeddings of the question
    df_copy = df.copy()
    df_copy["distances"] = distances_from_embeddings(
        question_embeddings,
        df_copy["embeddings"].values,
        distance_metric="cosine"
    )

    # Sort the copied dataframe by the distances and return it
    # (shorter distance = more relevant so we sort in ascending order)
    df_copy.sort_values("distances", ascending=True, inplace=True)
    return df_copy


# In[12]:


import tiktoken


def create_prompt(question, df, max_token_count):
    """
    Given a question and a dataframe containing rows of text and their
    embeddings, return a text prompt to send to a Completion model
    """
    # Create a tokenizer that is designed to align with our embeddings
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Count the number of tokens in the prompt template and question
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""

    current_token_count = len(tokenizer.encode(prompt_template)) + \
                          len(tokenizer.encode(question))

    context = []
    for text in get_rows_sorted_by_relevance(question, df)["text"].values:

        # Increase the counter based on the number of tokens in this row
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count

        # Add the row of text to the list if we haven't exceeded the max
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)


COMPLETION_MODEL_NAME = "gpt-3.5-turbo-instruct"


def answer_question(
        question, df, max_prompt_tokens=1800, max_answer_tokens=150
):
    """
    Given a question, a dataframe containing rows of text, and a maximum
    number of desired tokens in the prompt and response, return the
    answer to the question according to an OpenAI Completion model

    If the model produces an error, return an empty string
    """

    prompt = create_prompt(question, df, max_prompt_tokens)

    try:
        response = openai.Completion.create(
            model=COMPLETION_MODEL_NAME,
            prompt=prompt,
            max_tokens=max_answer_tokens
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

question1 = answer_question("Explain the fashion trends in 2023?", df)
print(question1)




