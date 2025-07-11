# Real Estate Agent - AI-Powered Real Estate Listings Generator

This project is a Python script that leverages OpenAI's GPT models and vector search to generate, parse, and semantically search real estate listings. It is designed to help real estate agents or platforms quickly create and search property listings tailored to user preferences.

## Features
- **AI-Generated Listings:** Uses OpenAI's GPT-3.5-turbo to generate realistic real estate listings with detailed descriptions.
- **Parsing & Validation:** Parses the AI-generated text into structured Python dictionaries and validates the data.
- **Semantic Search:** Uses sentence-transformers to embed both listings and user preferences, enabling semantic search for the best property matches.
- **Vector Database:** Stores listings and their embeddings in LanceDB for fast similarity search.
- **LLM-Powered Description Augmentation:** Optionally rewrites property descriptions to highlight features matching a buyer's preferences.

## How It Works
1. **Generate Listings:**
   - The script sends a prompt to OpenAI's API to generate 10 unique real estate listings.
2. **Parse Listings:**
   - The response is parsed using regex into a list of dictionaries, each representing a property.
3. **Validate Listings:**
   - Listings are validated and cleaned for consistency.
4. **Embed Listings:**
   - Each listing's description is embedded using a sentence-transformer model.
5. **Store in LanceDB:**
   - Listings and their embeddings are stored in a LanceDB table for efficient vector search.
6. **Semantic Search:**
   - User preferences are embedded and used to search for the most similar listings.
7. **Augment Descriptions:**
   - Optionally, the script can use the LLM to rewrite property descriptions to better match buyer preferences.

## Requirements
- Python 3.8+
- Packages:
  - openai
  - requests
  - lancedb
  - pyarrow
  - sentence-transformers
  - pydantic

Install dependencies with:
```bash
pip install openai requests lancedb pyarrow sentence-transformers pydantic
```

## Usage
1. **Set your OpenAI API key**
2. **Run the script:**
   ```bash
   python real-estate-agent.py
   ```
3. **View Results:**
   - The script prints generated listings, parses and validates them, stores them in LanceDB, and demonstrates semantic search and description augmentation.

## Customization
- **Prompt:** You can modify the prompt to generate different types of listings or change the format.
- **Search:** Change the `user_prompt` variable to search for different buyer preferences.

## Example Output
The script will print the top matching listings for a given buyer's preferences, including both the original and LLM-augmented descriptions.

---

**Note:** This script is for demonstration and prototyping purposes. For production use, ensure proper API key management and error handling.

