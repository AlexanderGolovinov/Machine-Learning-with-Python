import openai
import os
import json
import requests
import re
import lancedb
from lancedb.pydantic import LanceModel, vector
import pyarrow as pa
from sentence_transformers import SentenceTransformer

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key
api_key = os.environ.get("API_KEY")

prompt = """
Generate 10 unique real estate listings. Each listing should include exactly the following information:
- Neighborhood
- Price
- Bedrooms
- Bathrooms
- House Size
- Description (A short, engaging description (2-3 sentences))
- Neighborhood Description (A short, engaging description (2-3 sentences))

Format each listing as a numbered list.
"""

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    api_key=api_key,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=2000,
    temperature=0.8
)
listings_response = response.choices[0].message.content
print(listings_response)

# 1. Parse LLM response into a list of dicts
def parse_listings(text):
    """
    Parse LLM response into a list of dicts with improved regex pattern
    and error handling
    """
    listings = []

    # Updated regex pattern to match the actual format
    # The pattern now looks for numbered entries followed by property details
    pattern = re.compile(
        r"(\d+)\.\s*\n"  # Number followed by dot and newline
        r"Neighborhood:\s*(.*?)\n"  # Neighborhood
        r"Price:\s*(.*?)\n"  # Price
        r"Bedrooms:\s*(.*?)\n"  # Bedrooms
        r"Bathrooms:\s*(.*?)\n"  # Bathrooms
        r"House Size:\s*(.*?)\n"  # House Size
        r"Description:\s*(.*?)\n"  # Description
        r"Neighborhood Description:\s*(.*?)(?=\n\d+\.|$)",  # Neighborhood Description
        re.DOTALL | re.MULTILINE
    )

    matches = pattern.finditer(text)

    for match in matches:
        try:
            listing = {
                "title": f"Property {match.group(1)}",  # Generate title from number
                "neighborhood": match.group(2).strip(),
                "price": match.group(3).strip(),
                "bedrooms": match.group(4).strip(),
                "bathrooms": match.group(5).strip(),
                "house_size": match.group(6).strip(),
                "description": match.group(7).strip(),
                "neighborhood_description": match.group(8).strip(),
            }
            listings.append(listing)
        except Exception as e:
            print(f"Error parsing listing {match.group(1)}: {e}")
            continue

    return listings


def validate_listings(listings):
    """
    Validate parsed listings and clean up data
    """
    validated_listings = []

    for i, listing in enumerate(listings):
        # Check if all required fields are present and not empty
        required_fields = ['neighborhood', 'price', 'bedrooms', 'bathrooms', 'house_size', 'description',
                           'neighborhood_description']

        if all(listing.get(field, '').strip() for field in required_fields):
            # Clean up price field - remove any extra formatting
            listing['price'] = re.sub(r'[^\d,.$]', '', listing['price'])

            # Standardize numeric fields
            listing['bedrooms'] = re.sub(r'[^\d.]', '', listing['bedrooms'])
            listing['bathrooms'] = re.sub(r'[^\d.]', '', listing['bathrooms'])

            validated_listings.append(listing)
        else:
            print(f"Skipping incomplete listing {i + 1}: {listing}")

    return validated_listings


# Parse and validate listings
try:
    listings = parse_listings(listings_response)
    print(f"Parsed {len(listings)} listings")

    # Validate listings
    listings = validate_listings(listings)
    print(f"Validated {len(listings)} listings")

    if not listings:
        print("No valid listings found. Check the LLM response format.")
        exit(1)

    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    listing_texts = [
        f"{l['description']} {l['neighborhood_description']}"
        for l in listings
    ]
    embeddings = model.encode(listing_texts)

    # Define LanceModel schema
    class Listing(LanceModel):
        title: str
        neighborhood: str
        price: str
        bedrooms: str
        bathrooms: str
        house_size: str
        description: str
        neighborhood_description: str
        vector: vector(embeddings.shape[1])


    # Prepare data for LanceDB
    data = []
    for listing, embedding in zip(listings, embeddings):
        data.append({
            "title": listing['title'],
            "neighborhood": listing['neighborhood'],
            "price": listing['price'],
            "bedrooms": listing['bedrooms'],
            "bathrooms": listing['bathrooms'],
            "house_size": listing['house_size'],
            "description": listing['description'],
            "neighborhood_description": listing['neighborhood_description'],
            "vector": embedding
        })

    # Store in LanceDB
    DB_PATH = "~/.lancedb"
    TABLE_NAME = "real_estate_listings"

    db = lancedb.connect(DB_PATH)

    # Drop existing table if it exists
    try:
        db.drop_table(TABLE_NAME)
    except:
        pass  # Table doesn't exist, that's fine

    # Create table with proper schema
    table = db.create_table(TABLE_NAME, data=data, schema=Listing.to_arrow_schema())

    print(f"Successfully stored {len(listings)} listings in LanceDB.")

    # Optional: Print first few listings for verification
    print("\nFirst few listings:")
    for i, listing in enumerate(listings[:2]):
        print(f"\nListing {i + 1}:")
        print(f"  Neighborhood: {listing['neighborhood']}")
        print(f"  Price: {listing['price']}")
        print(f"  Bedrooms: {listing['bedrooms']}")
        print(f"  Bathrooms: {listing['bathrooms']}")
        print(f"  Description: {listing['description'][:100]}...")

except Exception as e:
    print(f"Error processing listings: {e}")
    import traceback
    traceback.print_exc()

listings = parse_listings(listings_response)

# --- Semantic Search Implementation ---
def embed_preferences(preferences: str, model=None):
    """Embed buyer preferences using the same embedding model as listings."""
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([preferences])[0]

def search_listings(preferences: str, top_k: int = 5):
    """Search LanceDB for listings most similar to buyer preferences."""
    # Reuse the embedding model
    pref_embedding = embed_preferences(preferences, model)
    db = lancedb.connect(DB_PATH)
    table = db.open_table(TABLE_NAME)
    results = table.search(pref_embedding).limit(top_k).to_pydantic(Listing)
    return results

# Example usage of the search function
user_prompt = "3 bedrooms, 2 bathrooms, modern kitchen, near downtown, budget $500,000"

if __name__ == "__main__":
    buyer_preferences = user_prompt
    matches = search_listings(buyer_preferences, top_k=3)
    print("\nTop matches for buyer preferences:")
    for m in matches:
        print(f"Title: {m.title}, Neighborhood: {m.neighborhood}, Price: {m.price}, Bedrooms: {m.bedrooms}, Bathrooms: {m.bathrooms}")
