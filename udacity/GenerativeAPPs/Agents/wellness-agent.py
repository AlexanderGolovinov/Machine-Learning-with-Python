import openai
import os

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key
api_key = os.environ.get("API_KEY")
openai.api_key = api_key

# Creating the prompt
user_prompt = f"How can I know my diet is improving my wellness?"
print(user_prompt)

# Function to call the OpenAI GPT-3.5 API
def wellness_agent(user_prompt):
    try:
        # Calling the OpenAI API with a system message and our prompt in the user message content
        # Use openai.ChatCompletion.create for openai < 1.0
        # openai.chat.completions.create for openai > 1.0
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
          {
            "role": "system",
            "content": """Your goal is to improve the wellness of the user by interleaving thought, action, and observation steps.
              (Thought Step) Begin by assessing the user's current wellness situation. Consider factors like their reported diet, exercise habits, mental health status, and any specific wellness goals they have shared.
              (Action Steps) Collect[Data from user] - Engage with the user to gather essential wellness information, data, or metrics. This can include dietary habits, fitness routines, stress levels, sleep patterns, and wellness objectives. 
                             Provide[Wellness Information] - Based on the collected data and current wellness trends, offer knowledge and insights about nutrition, exercise regimes, mental wellness practices, and relevant biological or medical information that supports and improves wellness. 
                             Recommend[Plan] - Conclude with a tailored recommendation or a specific action plan that the user can implement to enhance their wellness. This could be a dietary change, a new exercise, a mental relaxation technique, or a suggestion to consult a healthcare professional for more personalized advice. 
              (Observation Step) Respond to the user with the Action Steps, and observe the user's response and engagement. Gauge their understanding and willingness to follow the suggestions. Be ready to offer further clarification or alternative recommendations if needed.
              Repeat these steps N times until the user's wellness has improved.
              Example: 
              [User Query] I'm feeling stressed and not sleeping well. What can I do to improve my sleep? 
              (Thought) User is experiencing stress and poor sleep, likely interconnected issues. 
              Collect[Details about user's current stressors and sleep habits], 
              Provide[Information on relaxation techniques and sleep hygiene practices]. 
              Recommend)[Plan] Consider trying meditation before bed and establishing a regular sleep schedule. 
              What are some current stressors in your life? How many hours of sleep do you get each night?
              Have you tried meditation before bed? Do you have a regular sleep schedule?
              Consider trying meditation before bed and establishing a regular sleep schedule.
              Let's create a plan to meditate for 10 minutes before bed each night this week.
              What are some other wellness goals you have or wellness issues you are experiencing?
              """
          },
          {
            "role": "user",
            "content": user_prompt
          }
          ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
        # The response is a JSON object containing more information than the response. We want to return only the message content
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"

# Running the wellness agent
run_wellness_agent = wellness_agent(user_prompt)

# Printing the output.
print("Wellness Agent Response: ")
print(run_wellness_agent)