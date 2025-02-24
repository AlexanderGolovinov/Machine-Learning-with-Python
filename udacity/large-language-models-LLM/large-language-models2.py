import requests

TOGETHER_API_KEY = '92b03e79e980f9f0018c9f75f4598b74ac4c13df5c250b0daeb7f8d15f402c1b'
ENDPOINT = 'https://api.together.xyz/inference'

# Decoding parameters
TEMPERATURE = 0.0
MAX_TOKENS = 512
TOP_P = 1.0
TOP_K = 50
REPETITION_PENALTY = 1.0

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def query_together_endpoint(prompt):
    response = requests.post(
        ENDPOINT,
        json={
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "messages": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": REPETITION_PENALTY,
            "stop": ["<|eot_id|>", "<|eom_id|>"],
            "stream": False,
        },
        headers={
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json",
        },
    ).json()
    return response['output']['choices'][0]['text']


#Helper functions
def query_model(user_prompt_input, system_prompt=None, trigger=None, verbose=True, **kwargs):
    # Format the user and system prompts
    system_prompt = system_prompt or "You are a helpful assistant."
    inst_prompt = f"{B_INST} {user_prompt_input} {E_INST}"

    # Add trigger if provided
    if trigger:
        inst_prompt = inst_prompt + trigger

    # Prepare the system and user messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_input},
    ]

    # Call the Together API with the messages
    generation = query_together_endpoint(messages)

    if verbose:
        print(f"*** System Prompt ***\n{system_prompt}")
        print(f"*** User Prompt ***\n{user_prompt_input}")
        print(f"*** Full Messages ***\n{messages}")
        print(f"*** Generation ***\n{generation}")

    return generation


ANSWER_STAGE = "Provide the direct answer to the user question."
REASONING_STAGE = "Describe the step by step reasoning to find the answer."

# System prompt can be constructed in two ways:
# 1) Answering the question first or
# 2) Providing the reasoning first

# Similar ablation performed in "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
# https://arxiv.org/pdf/2201.11903.pdf
SYSTEM_PROMPT_TEMPLATE = """{b_sys}Answer the user's question using the following format:
1) {stage_1}
2) {stage_2}{e_sys}"""


# Chain of thought trigger from "Large Language Models are Zero-Shot Reasoners"
# https://arxiv.org/abs/2205.11916
COT_TRIGGER = "\n\nA: Lets think step by step:"
A_TRIGGER = "\n\nA:"

user_prompt_template = "Q: Llama 2 has a context window of {atten_window} tokens. \
If we are reserving {max_token} of them for the LLM response, \
the system prompt uses {sys_prompt_len}, \
the chain of thought trigger uses only {trigger_len}, \
and finally the conversational history uses {convo_history_len}, \
how many can we use for the user prompt?"

atten_window = 4096
max_token = 512
sys_prompt_len = 124
trigger_len = 11
convo_history_len = 390

user_prompt = user_prompt_template.format(
    atten_window=atten_window,
    max_token=max_token,
    sys_prompt_len=sys_prompt_len,
    trigger_len=trigger_len,
    convo_history_len=convo_history_len
)


desired_numeric_answer = atten_window - max_token - sys_prompt_len - trigger_len - convo_history_len
desired_numeric_answer

r = query_model(user_prompt_input=user_prompt)

system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
    b_sys = B_SYS,
    stage_1=ANSWER_STAGE,
    stage_2=REASONING_STAGE,
    e_sys=E_SYS
)

r2 = query_model(user_prompt_input=user_prompt, system_prompt=system_prompt)