from together import Together

client = Together(api_key='***')

response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "What are some fun things to do in Tallinn?"}],
)
print(response.choices[0].message.content)