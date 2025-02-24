import openai
import time

openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = ''

SYSTEM_PROMPT = """You expert at games of chance.
End every response with double exclamation points!!"""

USER_NAME = "User"
AI_NAME = "AI Assistant"
NEW_INTERACTION_DELIMITER = "\n\n"

# Decoding parameters
TEMPERATURE = 0.0
MAX_TOKENS = 3950  # Increased to simulate LLM with smaller attention window
TOP_P = 1.0

def query_openai(prompt):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
    )
    time.sleep(5) # to avoid rate limit errors
    if "error" in response:
        raise openai.InvalidRequestError(response["error"], param=None)
    else:
        return response.choices[0].text.strip().strip("\n")


def get_system_prompt(input_str=SYSTEM_PROMPT):
    return [f"System:{input_str}"]


def get_convo(input_str, convo):
    if not convo:
        convo = get_system_prompt()
    user_input_str = f"{USER_NAME}: {input_str}"
    response_trigger = f"{AI_NAME}: "
    convo.extend([user_input_str, response_trigger])
    return convo


# This is the function we will be fixing
def get_response(input_str, convo, use_simple_truncation, verbose):
    """
    Generate a response from an LLM based on user input_str and conversation history.

    Parameters:
    input_str (str): The user's current input_str or query to the language model.
    convo (list of str): A list representing the history of the conversation.
    use_simple_truncation (bool): A flag to determine whether to use a simple truncation
                                  method for managing conversation length.
    verbose (bool): A flag to determine if entire convo history should be printed.

    Returns:
    str: The generated response from the language model based on the current input_str and
         the conversation history.
    """
    convo = get_convo(input_str, convo)

    # Try to prompt model and catch if the prompt exceeds the attention window
    first_try = True
    atten_window_all_used_up = False
    while first_try or atten_window_all_used_up:
        # Convo list flattened into string to feed to model
        flattened_convo = NEW_INTERACTION_DELIMITER.join(convo) # SOLUTION HERE

        try:
            first_try = False
            response = query_openai(flattened_convo)
            atten_window_all_used_up = False

        except openai.InvalidRequestError as e:
            atten_window_all_used_up = True
            if verbose:
                print("** ATTEN_WINDOW ALL USED UP **")
                print(f"OpenAI Error: {repr(e)}\n")

            if not convo:
                return [
                    "Our Error: System prompt is using up too many tokens of the attention window"
                ]

            # We can recover from over-allocation of atten_window by removing components from history.
            if use_simple_truncation:
                # Just remove oldest element in convo
                convo = convo[1:]  # SOLUTION HERE

            else:
                # Remove the oldest User or AI convo turn, while retaining system prompt
                convo = convo[:1] + convo[2:]  # SOLUTION HERE

    # Add the LLM response to the response_trigger
    convo[-1] += response
    if verbose:
        print(NEW_INTERACTION_DELIMITER.join(convo))
    else:
        print(f"{USER_NAME}: {input_str}")
        print(f"{AI_NAME}: {response}")

    return convo


def chat(user_query, convo=[], use_simple_truncation=False, verbose=False):
    convo = get_response(user_query, convo, use_simple_truncation, verbose)
    return convo

user_inputs = [
    "What cards game has the best odds of winning?",
    "What are the odds of winning it?",
    "What is the best hand to be dealt?",
    "What is the next most likely game to win?",
]

convo = []
verbose = False
simple_truncation = True
for i, input in enumerate(user_inputs):
    print(f"**** Convo turn {i} ****")
    convo = chat(
        input, convo=convo, use_simple_truncation=simple_truncation, verbose=verbose
    )
    print()