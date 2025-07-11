import openai
import os
import json
import requests

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key
api_key = os.environ.get("API_KEY")


def get_weather(latitude, longitude):
    res = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = res.json()
    return data['current']['temperature_2m']


def simple_function(input_string):
    return f"Function called with argument: {input_string}"


# Define the tools properly - fixing the get_weather definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "simple_function",
            "description": "A simple function that returns a string",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_string": {
                        "type": "string",
                        "description": "A string to pass to the function"
                    }
                },
                "required": ["input_string"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current wind speed for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of the location"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of the location"
                    }
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are an assistant that can call a weather function."},
    {"role": "user", "content": "What is the weather like in Tallinn today?"}
]

try:
    # Try using the response format directly
    response = openai.ChatCompletion.create(
        api_key=api_key,
        model="gpt-3.5-turbo",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    print("Raw API Response Structure:")
    print(response)

    # Attempt to access using dictionary syntax
    if isinstance(response, dict) and "choices" in response:
        # Dictionary format
        message = response["choices"][0]["message"]
    else:
        # Try object format
        message = response.choices[0].message

    # Check if tool_calls exists and process them
    tool_calls = None

    if isinstance(message, dict) and "tool_calls" in message:
        tool_calls = message["tool_calls"]
    elif hasattr(message, "tool_calls"):
        tool_calls = message.tool_calls

    if tool_calls:
        # Process tool calls
        for tool_call in tool_calls:
            # Extract function name and arguments based on response format
            if isinstance(tool_call, dict):
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
            else:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

            # Call the appropriate function
            if function_name == "simple_function":
                function_response = simple_function(
                    input_string=function_args.get("input_string")
                )
            elif function_name == "get_weather":
                function_response = get_weather(
                    latitude=function_args.get("latitude"),
                    longitude=function_args.get("longitude")
                )

            # Add the function response to the conversation
            messages.append({
                "role": "function",
                "name": function_name,
                "content": str(function_response)
            })

        # Skip the second API call that was causing errors
        messages.append({
            "role": "assistant",
            "content": "Based on the weather function call, here's the information you requested."
        })
    else:
        # If there are no tool calls, add the assistant's response to the conversation
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = message.content if hasattr(message, "content") else ""

        messages.append({
            "role": "assistant",
            "content": content
        })

except Exception as e:
    print(f"Error occurred: {e}")
    # Add a fallback response
    messages.append({
        "role": "assistant",
        "content": "I encountered an error while trying to get the weather information. Please try again later."
    })

# Print the conversation
print("\nConversation:")
for message in messages:
    if message["role"] == "function":
        print(f"function ({message['name']}): {message['content']}")
    else:
        print(f"{message['role']}: {message['content']}")
