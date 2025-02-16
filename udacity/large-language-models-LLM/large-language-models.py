from together import Together

together_api_key = ''
client = Together(api_key=together_api_key)

def get_together_response(message):
    return client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": message}],
)

response = get_together_response("What are some fun things to do in New York?")

print(response.choices[0].message.content)