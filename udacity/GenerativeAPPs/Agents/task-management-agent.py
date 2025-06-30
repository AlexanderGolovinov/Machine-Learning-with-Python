import openai
openai.api_base = "https://openai.vocareum.com/v1"
import os
import json
import pandas as pd

openai.api_base = "https://openai.vocareum.com/v1"

# Define OpenAI API key
api_key = os.environ.get("API_KEY")
openai.api_key = api_key

# Load the project management data
df = pd.read_csv('project_management.csv')

def task_retrieval_and_status_updates(task_id, status, last_updated):
    """Retrieve and update task status"""
    df.loc[df['Task ID'] == task_id, 'Status'] = status
    df.loc[df['Task ID'] == task_id, 'Last Updated'] = last_updated
    df.to_csv('project_management.csv', index=False)  # save changes to file
    task = df.loc[df['Task ID'] == task_id]
    return json.dumps(task.to_dict())

def project_reporting_and_analytics(project_id):
    """Generate reports on project progress and team performance"""
    project = df.loc[df['Project ID'] == project_id]
    return json.dumps(project.to_dict())

def resource_allocation_and_scheduling(task_id, assigned_to, time_estimate, due_date, status):
    """Allocate tasks based on current workloads and schedules"""
    df.loc[df['Task ID'] == task_id, 'Assigned To'] = assigned_to
    df.loc[df['Task ID'] == task_id, 'Time Estimate'] = time_estimate
    df.loc[df['Task ID'] == task_id, 'Due Date'] = due_date
    df.loc[df['Task ID'] == task_id, 'Status'] = status
    df.to_csv('project_management.csv', index=False)  # save changes to file
    return json.dumps(task.to_dict())

def run_conversation():
    # messages is a list of initial conversation messages. The system message describes the role of the assistant. The second message is from the user, the user prompt
    messages = [
        {"role": "system", "content": "You are a project management assistant with knowledge of project statuses, task assignments, and scheduling. You can provide updates on projects, assign tasks to team members, and schedule meetings. You understand project management terminology and are capable of parsing detailed project data. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."
 },
        {"role": "user", "content": "Change the status of task 2 to completed."} # this prompt should call task_retrieval_and_status_updates
    ]
    # tools is a list of functions that the assistant can use. Each function is described by its name, description, and parameters.
    tools= [
        {
            "type": "function",
            "function": {
                "name": "task_retrieval_and_status_updates",
                "description": "Retrieve and update task status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "integer",
                            "description": "The unique identifier for the task"
                        },
                        "status": {
                            "type": "string",
                            "description": "The new status of the task"
                        },
                        "last_updated": {
                            "type": "string",
                            "description": "The date of the last status update or change to the task"
                        }
                    },
                    "required": ["task_id", "status", "last_updated"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "project_reporting_and_analytics",
                "description": "Generate reports on project progress and team performance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "integer",
                            "description": "The unique identifier for the project"
                        }
                    },
                    "required": ["project_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "resource_allocation_and_scheduling",
                "description": "Allocate tasks based on current workloads and schedules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "integer",
                            "description": "The unique identifier for the task"
                        },
                        "assigned_to": {
                            "type": "string",
                            "description": "The user ID or name of the person to whom the task is assigned"
                        },
                        "time_estimate": {
                            "type": "integer",
                            "description": "An estimate of the time required to complete the task"
                        },
                        "due_date": {
                            "type": "string",
                            "description": "The deadline for the task completion"
                        },
                        "status": {
                            "type": "string",
                            "description": "The current status of the task"
                        }
                    },
                    "required": ["task_id", "assigned_to", "time_estimate", "due_date", "status"]
                }
            }
        }
    ]
    # `openai.chat.completions.create()` is called to generate a response from the GPT-3 model. The model, messages, and tools are passed as arguments. The `tool_choice` is set to "auto", allowing the model to choose which tool (function) to use.
    # Use openai.ChatCompletion.create for openai < 1.0
    # openai.chat.completions.create for openai > 1.0
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # let the model decide which tool (function) to use
    )
    # response_message and tool_calls extract the first response message and any tool calls from the response.
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls # get the tool calls from the first response
    print(tool_calls)
    # end of first response, now we parse the response and call the functions the model identified from our tool list
    # check if the model wanted to call a function
    if tool_calls:
        # list the available functions and their corresponding python functions
        available_functions = {
            "task_retrieval_and_status_updates": task_retrieval_and_status_updates,
            "project_reporting_and_analytics": project_reporting_and_analytics,
            "resource_allocation_and_scheduling": resource_allocation_and_scheduling,
        }
        messages.append(response_message)  # extend the conversation with the first response
        # send the info for each function call and function response to the model
        for tool_call in tool_calls: # iterate through the tool calls in the response
            function_name = tool_call.function.name # get the name of the function to call
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments) # converting the arguments of the function call from a JSON formatted string into a Python dictionary.
            if function_name == 'task_retrieval_and_status_updates':
                function_response = function_to_call( # call the function with the arguments. The result of the function call is stored in function_response
                    task_id=function_args.get("task_id"),
                    status=function_args.get("status"),
                    last_updated=function_args.get("last_updated")
                )
            elif function_name == 'project_reporting_and_analytics':
                function_response = function_to_call(
                    project_id=function_args.get("project_id")
                )
            elif function_name == 'resource_allocation_and_scheduling':
                function_response = function_to_call(
                    task_id=function_args.get("task_id"),
                    assigned_to=function_args.get("assigned_to"),
                    time_estimate=function_args.get("time_estimate"),
                    due_date=function_args.get("due_date"),
                    status=function_args.get("status")
                )

            message_to_append = {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response, # send the function response to the model, it's the JSON string of the function response
                }
            messages.append(message_to_append)  # extend conversation with function response

        # See https://gist.github.com/gaborcselle/2dc076eae23bd219ff707b954c890cd7
        #messages[1].content = "" # clear the first message (parsing bug)
        messages[1]['content'] = "" # clear the first message (parsing bug)

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response

print(run_conversation()) # will print the second response from the model