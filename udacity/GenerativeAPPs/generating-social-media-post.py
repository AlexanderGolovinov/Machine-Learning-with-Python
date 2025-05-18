import openai
import os
openai.api_base = "https://openai.vocareum.com/v1"

api_key = os.environ.get("API_KEY")
openai.api_key = api_key

# Mocked hardcoded user input data
product_name = "Pet sleeping bag"
cool_feature = "Warm and cozy donut sleeping bag"
audience_persona = "pet homeowners"

prompt = f"Create a catchy, clever 140 character social media post targeted toward {audience_persona} introducing and promoting a new {cool_feature} feature of {product_name}."
print(prompt)

# Function to call the OpenAI GPT-3.5 API
def generate_social_media_post(prompt):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        # Use openai.ChatCompletion.create for openai < 1.0
        # openai.chat.completions.create for openai > 1.0
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
          {
            "role": "system",
            "content": "You are a social media influencer and writer. "
          },
          {
            "role": "user",
            "content": prompt
          }
          ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Generating the social media post
generated_post = generate_social_media_post(prompt)

# Printing the output.
print("Generated Social Media Post:")
print(generated_post)