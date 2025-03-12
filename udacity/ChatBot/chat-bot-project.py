import pandas as pd
import numpy as np
import re
from openai import OpenAI
import os
from sklearn.model_selection import train_test_split

# import kagglehub
# path = kagglehub.dataset_download("/data/Reviews.csv")
# print("Path to dataset files:", path)

df = pd.read_csv('../../data/Reviews.csv')

# Take a look at the first few rows
print(f"Dataset shape: {df.shape}")
df.head()

# Clean the text data
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
df['text'] = df['Text'].apply(clean_text)

# Filter out empty reviews
df = df[df['text'].str.len() > 0]

# Sample 1000 reviews for our chatbot dataset
chatbot_data = df.sample(1000, random_state=42)

# Verify we have at least 20 rows
print(f"Final dataset shape: {chatbot_data.shape}")
chatbot_data.head()

chatbot_data.to_csv('food_reviews_cleaned.csv', index=False)

# Create a smaller subset with just the columns we need
chatbot_subset = chatbot_data[['ProductId', 'Score', 'text']].copy()

# Verify our final dataset
print(f"Number of unique products: {chatbot_subset['ProductId'].nunique()}")
print(f"Average review score: {chatbot_subset['Score'].mean():.2f}")
chatbot_subset.head()

base_url = "https://openai.vocareum.com/v1"
api_key = os.environ.get("OPEN_AI_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

# Define a function to create embeddings for our text data
def get_embeddings(texts, model="text-embedding-ada-002"):
    embeddings = []
    batch_size = 20

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Create embeddings for the batch
            response = client.embeddings.create(
                input=batch,
                model=model
            )
            # Extract embeddings from response and add to our list
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating embeddings for batch {i // batch_size}: {e}")
            # Return empty embeddings for failed batch items
            embeddings.extend([[] for _ in range(len(batch))])

    return embeddings

# Generate embeddings for a subset of our data
# This might take some time for larger datasets

sample_size = 100  # Adjust based on your API usage limits
sample_data = chatbot_subset.sample(sample_size, random_state=42)

# Get embeddings
sample_data['embedding'] = get_embeddings(sample_data['text'].tolist())

# Save embeddings to avoid regenerating them
import pickle
with open('food_review_embeddings.pkl', 'wb') as f:
    pickle.dump(sample_data, f)

# Define a function for semantic search
from scipy.spatial.distance import cosine

def find_similar_reviews(query, df_with_embeddings, top_n=5):
    # Get the embedding for the query
    query_embedding = get_embeddings([query])[0]

    # Calculate similarities
    similarities = []
    for idx, row in df_with_embeddings.iterrows():
        similarity = 1 - cosine(query_embedding, row['embedding'])
        similarities.append((idx, similarity))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return top N similar reviews
    top_indices = [idx for idx, _ in similarities[:top_n]]
    return df_with_embeddings.loc[top_indices]


# Define our custom query function
def custom_food_review_query(query, sample_data, model="gpt-3.5-turbo"):
    # Find relevant reviews
    relevant_reviews = find_similar_reviews(query, sample_data)

    # Create a context from the relevant reviews
    context = "Here are some relevant food reviews:\n\n"
    for _, row in relevant_reviews.iterrows():
        context += f"Product ID: {row['ProductId']}\n"
        context += f"Score: {row['Score']} / 5\n"
        context += f"Review: {row['text']}\n\n"

    # Create the prompt
    prompt = f"""Based on the following food product reviews, please answer this query:

{query}

{context}

Please provide a helpful response based only on the information in these reviews.
"""

    # Get the completion
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant specializing in food product information based on customer reviews."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Question: " + query)
    return response.choices[0].message.content


# Load our saved embeddings
with open('food_review_embeddings.pkl', 'rb') as f:
    sample_data = pickle.load(f)

response = ""

response = custom_food_review_query("What are some highly rated chocolate products?", sample_data)
print(response)

response = custom_food_review_query("What products are the healthiest ones?", sample_data)
print(response)