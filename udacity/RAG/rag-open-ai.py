import openai
import os

openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = os.environ.get("OPEN_AI_KEY")

import requests
import pandas as pd
from dateutil.parser import parse


# Get the Wikipedia page for the 2023 Turkey–Syria earthquake
params = {
    "action": "query",
    "prop": "extracts",
    "exlimit": 1,
    "titles": "2023_Turkey–Syria_earthquakes",
    "explaintext": 1,
    "formatversion": 2,
    "format": "json"
}
resp = requests.get("https://en.wikipedia.org/w/api.php", params=params)
response_dict = resp.json()

# Load page text into a dataframe
df = pd.DataFrame()
df["text"] = response_dict["query"]["pages"][0]["extract"].split("\n")

# Clean up dataframe to remove empty lines and headings
df = df[(
    (df["text"].str.len() > 0) & (~df["text"].str.startswith("=="))
)].reset_index(drop=True)
df.head()

EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
response = openai.Embedding.create(
    input=[df["text"][0]],
    engine=EMBEDDING_MODEL_NAME
)

# Extract and print the first 20 numbers in the embedding
response_list = response["data"]
first_item = response_list[0]
first_item_embedding = first_item["embedding"]
print(first_item_embedding[:20])

# Send text data to the model
response = openai.Embedding.create(
    input=df["text"].tolist(),
    engine=EMBEDDING_MODEL_NAME
)

# Extract embeddings
embeddings = [data["embedding"] for data in response["data"]]

# Add embeddings list to dataframe
df["embeddings"] = embeddings
df.to_csv("embeddings.csv")
