from lancedb.pydantic import vector, LanceModel


class CatsAndDogs(LanceModel):
    vector: vector(2)
    species: str
    breed: str
    weight: float


import lancedb

db = lancedb.connect("~/.lancedb")
table_name = "cats_and_dogs"
db.drop_table(table_name, ignore_missing=True)
table = db.create_table(table_name, schema=CatsAndDogs)

data = [
    CatsAndDogs(
        vector=[1., 0.],
        species="cat",
        breed="shorthair",
        weight=12.,
    ),
    CatsAndDogs(
        vector=[-1., 0.],
        species="cat",
        breed="himalayan",
        weight=9.5,
    ),
    CatsAndDogs(
        vector=[0., 10.],
        species="dog",
        breed="samoyed",
        weight=47.5,
    ),
    CatsAndDogs(
        vector=[0, -1.],
        species="dog",
        breed="corgi",
        weight=26.,
    )
]

table.add([dict(d) for d in data])
print(table.head().to_pandas())

res = table.search([10.5, 10.,]).limit(1).to_pandas()
print(res)


from lance.vector import vec_to_table
import numpy as np

mat = np.random.randn(100_000, 16)
table_name = "exercise3_ann"
db.drop_table(table_name, ignore_missing=True)
table = db.create_table(table_name, vec_to_table(mat))

query = np.random.randn(16)
res = table.search(query).limit(10).to_pandas()
print(res)

table = db["cats_and_dogs"]
table = db["exercise3_ann"]

print(table.list_versions())