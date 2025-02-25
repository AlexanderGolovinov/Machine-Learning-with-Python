# Natural Language Models
from transformers import AutoTokenizer

# pretrained tokenizer to use from HuggingFace
my_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Simple method getting tokens from text
raw_text = '''Rory's shoes are magenta and so are Corey's but they aren't nearly as dark!'''
tokens = my_tokenizer.tokenize(raw_text)

print(tokens)

# This method also returns special tokens depending on the pretrained tokenizer
detailed_tokens = my_tokenizer(raw_text).tokens()

print(detailed_tokens)

# Tokenizer method to get the IDs if we already have the tokens as strings
detailed_ids = my_tokenizer.convert_tokens_to_ids(detailed_tokens)
print(detailed_ids)

# Integer IDs for tokens
ids = my_tokenizer.encode(raw_text)

# The inverse of the .enocde() method: .decode()
my_tokenizer.decode(ids)
# To ignore special tokens (depending on pretrained tokenizer)
my_tokenizer.decode(ids, skip_special_tokens=True)
# List of tokens as strings instead of one long string
my_tokenizer.convert_ids_to_tokens(ids)

phrase = '🥱 the dog next door kept barking all night!!'
ids = my_tokenizer.encode(phrase)
print(phrase)
print(my_tokenizer.convert_ids_to_tokens(ids))
print(my_tokenizer.decode(ids))

phrase = '''wow my dad thought mcdonalds sold tacos \N{SKULL}'''
ids = my_tokenizer.encode(phrase)
print(phrase)
print(my_tokenizer.convert_ids_to_tokens(ids))
print(my_tokenizer.decode(ids))