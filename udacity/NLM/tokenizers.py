from __future__ import annotations

# COMPLETE: Feel free to add other imports as needed
import string
import re
from collections import defaultdict

sample_text = '''Mr. Louis continued to say, "Penguins are important, 
but we mustn't forget the nuumber 1 priority: the READER!"
'''

print(sample_text)

def normalize_text(text: str) -> str:
    # COMPLETE: Normalize incoming text; can be multiple actions
    # Only keep ASCII letters, numbers, punctuation, and whitespace characters
    acceptable_characters = (
        string.ascii_letters
        + string.digits
        + string.punctuation
        + string.whitespace
    )
    normalized_text = ''.join(
        filter(lambda letter: letter in acceptable_characters, text)
    )
    # Make text lower-case
    normalized_text = normalized_text.lower()
    return normalized_text

def pretokenize_text(text: str) -> list[str]:
    # COMPLETE: Pretokenize normalized text
    # Split based on spaces
    smaller_pieces = text.split()
    return smaller_pieces

# Test out your pretokenization step (after normalizing the text)
normalized_text = normalize_text(sample_text)
pretokenize_text(normalized_text)

# Test out your normalization
normalize_text(sample_text)

# Combine normalization and pretokenization steps before breaking things further
def tokenize_text(text: str) -> list[str]:
    # Apply created steps
    normalized_text: str = normalize_text(text)
    pretokenized_text: list[str] = pretokenize_text(normalized_text)
    # COMPLETE: Go through pretokenized text to create a list of tokens
    tokens = []
    # Small 'pieces' to make full tokens
    for word in pretokenized_text:
        tokens.extend(
            re.findall(
                f'[\w]+|[{string.punctuation}]', # Split word at punctuations
                word,
            )
        )
    return tokens

# Test out your tokenization (that uses normalizing & pretokenizing functions)
tokenize_text(sample_text)

# Useful for some tasks
def postprocess_tokens(tokens: list[str]) -> list[str]:
    # COMPLETE: Add beginning and end of sequence tokens to your tokenized text
    # Can use a format like '[BOS]' & '[EOS]'
    bos_token = '[BOS]'
    eos_token = '[EOS]'
    updated_tokens = (
        [bos_token]
        + tokens
        + [eos_token]
    )
    return updated_tokens

# Test full pipeline (normalizing, pretokenizing, tokenizing, & postprocessing)
tokens = tokenize_text(sample_text)
tokens = postprocess_tokens(tokens)

print(tokens)

# Sample corpus (normally this would be much bigger)
sample_corpus = (
    '''Mr. Louis continued to say, "Penguins are important, \nbut we mustn't forget the nuumber 1 priority: the READER!"''',
    '''BRUTUS:\nHe's a lamb indeed, that baes like a bear.''',
    '''Both by myself and many other friends:\mBut he, his own affections' counsellor,\nIs to himself--I will not say how true--\nBut to himself so secret and so close,'''
)

# Retrieve unique tokens (from the pipeline defined above) in a set
unique_tokens = set()
for text in sample_corpus:
    tokens_from_text = tokenize_text(text)
    tokens_from_text = postprocess_tokens(tokens_from_text)
    unique_tokens.update(tokens_from_text)

# Create mapping (dictionary) for unique tokens using arbitrary & unique IDs
token2id = defaultdict(lambda : 0) # Allow for unknown tokens to map to 0
token2id |= {
    token: idx
    for idx, token in enumerate(unique_tokens, 1) # Skip 0 (represents unknown)
}

# A mapping for IDs to convert back to token
id2token = defaultdict(lambda : '[UNK]') # Allow for unknown token ('[UNK]')
id2token |= {
    idx: token
    for token, idx in token2id.items()
}


def encode(tokens: list[str]) -> list[int]:
    # COMPLETE: Complete this function to encode tokens to integer IDs
    encoded_tokens = [token2id[token] for token in tokens]
    return encoded_tokens

# Use sample text for testing
sample_text = sample_corpus[0]
# Create tokens (to be fed to encode())
tokens = tokenize_text(sample_text)
tokens = postprocess_tokens(tokens)
print(f'Tokens:\n{tokens}\n')

# Test encode()
encoded_tokens = encode(tokens)
print(f'Encoded Tokens:\n{encoded_tokens}\n')

# COMPLETE: Create an encoder to transform IDs (from encode()) to token strings

def decode(ids: list[int]) -> list[str]:
    # COMPLETE: Complete this function to decode integer IDs to token strings
    token_strings = [id2token[idx] for idx in ids]
    return token_strings

# Use sample text for testing
sample_text = sample_corpus[0]
# Create tokens
tokens = tokenize_text(sample_text)
tokens = postprocess_tokens(tokens)
print(f'Tokens:\n{tokens}\n')

# Create token IDs (to be fed to decode())
encoded_tokens = encode(tokens)
print(f'Encoded Tokens:\n{encoded_tokens}\n')

# Test out decode()
decoded_tokens = decode(encoded_tokens)
print(f'Decoded Tokens:\n{decoded_tokens}\n')