import lancedb

import numpy as np
import pandas as pd

ratings = pd.read_csv('../../data/ratings.csv', header=0)
ratings

reviewmatrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
reviewmatrix

matrix = reviewmatrix.values
_, _, vh = np.linalg.svd(matrix, full_matrices=False)
embeddings = vh.T

movies = pd.read_csv('../../data/movies.csv', header=0)
movies = movies.set_index("movieId").reindex(reviewmatrix.columns)
movies

links = pd.read_csv('../../data/links.csv', header=0)
links = links.set_index("movieId").reindex(reviewmatrix.columns)
links

from lancedb.pydantic import vector, LanceModel


class Content(LanceModel):
    movie_id: int
    vector: vector(embeddings.shape[1])
    genres: str
    title: str
    imdb_id: int

    @property
    def imdb_url(self) -> str:
        return f"https://www.imdb.com/title/tt{self.imdb_id}"

values = list(zip(*[reviewmatrix.columns,
                    embeddings,
                    movies["genres"],
                    movies["title"],
                    links["imdbId"],
                    links["tmdbId"]]))
keys = Content.__annotations__.keys()
data = [dict(zip(keys, v)) for v in values]

print(data[0])

import pyarrow as pa
db = lancedb.connect("~/.lancedb")
table_name = "movielens_small"
db.drop_table(table_name, ignore_missing=True)
data = pa.Table.from_pylist(data, schema=Content.to_arrow_schema())
table = db.create_table(table_name, data=data)

def get_recommendations(title: str) -> list[(int, str, str)]:
    # First we retrieve the vector for the input title
    query_vector = (table.to_lance()
                    .to_table(filter=f"title='{title}'")["vector"].to_numpy()[0])
    # Please write the code to search for the 5 most similar titles
    results = table.search(query_vector).limit(5).to_pydantic(Content)
    # For each result, return the movie_id, title, and imdb_url
    return [(c.movie_id, c.title, c.imdb_url) for c in results]

recommendations = get_recommendations("Moana (2016)")
print(recommendations)

recommendations = get_recommendations("Rogue One: A Star Wars Story (2016)")
print(recommendations)