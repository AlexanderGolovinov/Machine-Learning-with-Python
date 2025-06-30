def read_quotes() -> list[str]:
    with open("rick_and_morty_quotes.txt", "r") as fh:
        return fh.readlines()

rick_and_morty_quotes = read_quotes()
rick_and_morty_quotes[:3]

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

emb1, emb2 = model.encode([
    "Losers look stuff up while the rest of us are carpin' all them diems.\n",
    "Losers look stuff up while the rest of us are carpin' all them diems."
])

np.allclose(emb1, emb2)

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Union

MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

def generate_embeddings(input_data: Union[str, list[str]]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(input_data)
    return embeddings

#Print the embeddings
for sentence, embedding in zip(rick_and_morty_quotes[:3], embeddings[:3]):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

len(embeddings[0])
np.linalg.norm(embeddings, axis=1)

query_text = "Are you the cause of your parents' misery?"
query_embedding = model.encode(query_text)


import numpy as np

def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the Euclidean distance between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.

    Returns
    -------
    float
        Euclidean distance between `v1` and `v2`.
    """
    dist = v1 - v2
    return np.linalg.norm(dist, axis=len(dist.shape)-1)


def find_nearest_neighbors(query: np.ndarray,
                           vectors: np.ndarray,
                           k: int = 1) -> np.ndarray:
    """
    Find k-nearest neighbors of a query vector.

    Parameters
    ----------
    query : np.ndarray
        Query vector.
    vectors : np.ndarray
        Vectors to search.
    k : int, optional
        Number of nearest neighbors to return, by default 1.

    Returns
    -------
    np.ndarray
        The `k` nearest neighbors of `query` in `vectors`.
    """
    distances = euclidean_distance(query, vectors)
    return np.argsort(distances)[:k]

indices = find_nearest_neighbors(query_embedding, embeddings, k=3)

for i in indices:
    print(rick_and_morty_quotes[i])