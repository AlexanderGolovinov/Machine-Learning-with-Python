import openai
import os
import json
import requests

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

print(response.choices[0].message.content)