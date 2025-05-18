import openai
import os
import json

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key
api_key = os.environ.get("API_KEY")
openai.api_key = api_key

import requests


def get_weather(latitude, longitude):
    res = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
    data = res.json()
    return data['current']['temperature_2m']


def simple_function(input_string):
    return f"Function called with argument: {input_string}"


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
                "required": ["input_string"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "name": "get_weather",
    #     "description": "Get current temperature for a given location.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "location": {
    #                 "type": "string",
    #                 "description": "City and country e.g. Bogotá, Colombia"
    #             }
    #         },
    #         "required": [
    #             "location"
    #         ],
    #         "additionalProperties": False
    #     }
    # }
]

messages = [
    {"role": "system", "content": "You are an assistant that can call a simple function."},
    {"role": "user", "content": "Call the simple function with the argument 'Hello there, World!'."}
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

tool_calls = response.choices[0].message.tool_calls

if tool_calls:
    available_functions = {
        "simple_function": simple_function,
        "get_weather": get_weather
    }

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)

        if function_name == 'simple_function':
            function_response = function_to_call(
                input_string=function_args.get("input_string"),
            )

        if function_name == 'get_weather':
            function_response = function_to_call(
                input_string=function_args.get("location"),
            )

        messages.append({
            "role": "assistant",
            "content": function_response,
        })

for message in messages:
    print(f"{message['role']}: {message['content']}")
